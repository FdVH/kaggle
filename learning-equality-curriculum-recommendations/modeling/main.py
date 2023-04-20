# --------------------------------------------------------------------------- #
#                                  Imports                                    #
# --------------------------------------------------------------------------- #
import os
import gc
import shutil
import time
import logging
import psutil
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_random_edge
from torch_geometric.utils.subgraph import bipartite_subgraph
from transformers import AutoModel,\
    AutoModelForSequenceClassification, AutoConfig,\
    PreTrainedTokenizerFast, BatchEncoding
from sklearn.model_selection import GroupKFold
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union, Optional
from functools import wraps
from datetime import datetime
from collections import defaultdict
from inspect import getframeinfo, stack
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                                 Utilities                                   #
# --------------------------------------------------------------------------- #
# Topic & content classes from competition hosts
class Topic:
    def __init__(self, topic_id):
        self.id = topic_id
        
    @property
    def parent(self):
        parent_id = topics_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)
        
    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors
    
    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]
        
    def get_breadcrumbs(self, separator=" >> ", include_self=True,
                        include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))
    
    @property
    def children(self):
        return [Topic(child_id) for child_id in topics_df[topics_df.parent
                                                          == self.id].index]
    
    def subtree_markdown(self, depth=0):
        markdown = "  " * depth + "- " + self.title + "\n"
        for child in self.children:
            markdown += child.subtree_markdown(depth=depth + 1)
        for content in self.content:
            markdown += ("  " * (depth + 1) + "- " + 
                "[" + content.kind.title() + "] " + content.title) + "\n"
        return markdown
    
    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id
    
    def __getattr__(self, name):
        return topics_df.loc[self.id][name]
    
    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"
    
    # NEW
    # Get all content items in the subtree of this topic
    @property
    def subtree_content(self):
        content = self.content
        for child in self.children:
            content += child.subtree_content
        return content
    
    # Get root topic of this topic
    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.ancestors[-1]
        
    # EDITED
    # Originally 'content' property, now method with option to get ids only.
    def get_content(self, ids=False):
        if self.id in correlations_df.index:
            if ids: return correlations_df.loc[self.id].content_ids
            return [ContentItem(content_id) for content_id in 
                    correlations_df.loc[self.id].content_ids]
        else:
            return tuple([]) if self.has_content else []
        
    @property
    def content(self):
        return self.get_content()
    
    @property
    def content_ids(self):
        return self.get_content(ids=True)
    
class ContentItem:
    def __init__(self, content_id):
        self.id = content_id
        
    def __getattr__(self, name):
        return content_df.loc[self.id][name]
    
    def __str__(self):
        return self.title
    
    def __repr__(self):
        return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"
    
    def __eq__(self, other):
        if not isinstance(other, ContentItem):
            return False
        return self.id == other.id
    
    def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
        breadcrumbs = []
        for topic in self.topics:
            new_breadcrumb = topic.get_breadcrumbs(separator=separator,
                                                   include_root=include_root)
            if new_breadcrumb:
                new_breadcrumb = new_breadcrumb + separator + self.title
            else:
                new_breadcrumb = self.title
            breadcrumbs.append(new_breadcrumb)
        return breadcrumbs
    
    # EDITED
    @property
    def topics(self):
        return [Topic(topic_id) for topic_id in correlations_df.mask(
            ~correlations_df.applymap(lambda x: self.id in x)).dropna().index]

# Setup functions
def set_seed(seed: int = 13) -> None:
    global g
    g_pt = torch.Generator()
    g_pt.manual_seed(seed)
    torch.manual_seed(seed)
    g_np = np.random.default_rng(seed)
    g = {'pt': g_pt, 'np': g_np}
    
def setup_output(checkpoints_dir: str, clear_checkpoints: bool = False,
                 clear_working: bool = False) -> None:
    # Clear and/or make checkpoints directory
    if clear_checkpoints and checkpoints_dir.is_dir():
        shutil.rmtree(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear working directory
    if clear_working:
        for path in Path('/kaggle/working').iterdir():
            if path.name.startswith('__'):
                continue
            if path.is_file():
                os.remove(path)
            elif path.is_dir():
                shutil.rmtree(path)

def get_logger(logfile: Path) -> logging.Logger:
    # Init logger
    if logfile.is_file(): os.remove(logfile)
    logfile.parents[0].mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(logfile)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)
    
    # Format handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

# Additional utilities
def print_log(logfile: Path) -> None:
    try:
        with open(logfile) as f:
            for line in f.readlines():
                print(line)
    except:
        pass
        
def timeit(func: Callable) -> Callable:
    """Debug decorator to time a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        logger.debug(f'{func.__name__} took {t1 - t0:.6f} s to complete.')
        return out
    return wrapper

def DEBUG_RAM() -> None:
    """Debug function to print RAM and VRAM usage."""
    caller = getframeinfo(stack()[1][0])
    try:
        gpu_ram = round(torch.cuda.memory_allocated() / 
                        torch.cuda.max_memory_allocated(), 1)
    except ZeroDivisionError:
        gpu_ram = None
    ram = psutil.virtual_memory().percent
    vram = round(psutil.virtual_memory().available * 100 /
                 psutil.virtual_memory().total, 1)
    logger.debug(f'{caller.filename}:{caller.lineno} - ' \
                 f'RAM {ram}% : VRAM {vram}% : GPU_RAM {gpu_ram}%')
        
def id_to_int(item_ids: Union[np.ndarray[str], pd.Index[str]], 
              item_type: str = 'topic') -> np.ndarray:
    """Get the integer representation of topic or content id(s)."""
    global topics_df, content_df
    if type(item_ids) == str:
        item_ids = [item_ids]
    if item_type == 'topic':
        return topics_df.loc[item_ids, 'num'].values
    elif item_type == 'content':  
        return content_df.loc[item_ids, 'num'].values
    else:
        raise Exception(f'item_type: {item_type} not implemented')
        
def int_to_id(item_nums: Union[List[int], np.ndarray[int]],
              item_type: str = 'topic') -> pd.Index:
    """Get the string id representation of topic or content number(s)."""
    if type(item_nums) == int:
        item_nums = [item_nums]
    if item_type == 'topic':
        return topics_df.index[item_nums]
    elif item_type == 'content':  
        return content_df.index[item_nums]
    else:
        raise Exception(f'item_type: {item_type} not implemented')

def tensor_dict_to(tensor_dict: Dict[Tensor], pin_memory: bool = False, 
                   device: Union[str, torch.device] = None) -> Dict[Tensor]:
    """Return a dictionary of tensors on the specified device, with option
    to stage in pinned memory."""
    out = {}
    for k, t in tensor_dict.items():
        if pin_memory and (device is not None):
            out[k] = t.pin_memory().to(device, non_blocking=True)
        elif pin_memory:
            out[k] = t.pin_memory()
        elif device is not None:
            out[k] = t.to(device)
        else:
            out[k] = t
    return out
        
def get_ids(item_list: List[Union[Topic, ContentItem]]) -> List[str]:
    idx = [item.id for item in item_list]
    assert len(item_list)==len(idx), "Input and output lengths do not match"
    return idx

# --------------------------------------------------------------------------- #
#                               Pre-processing                                #
# --------------------------------------------------------------------------- #
def make_repr(df: pd.DataFrame, use_title: bool = True, use_descr: bool = False,
              use_text: bool = False, use_level: bool = False) -> List[str]:
    """Return a list of concatenated text representations for all topic or
    content items with text features in a DataFrame."""
    fields = []
    if use_title: fields.append('title')
    if use_descr: fields.append('description')
    if use_text: fields.append('text')
    if use_level: fields.append('level')
    
    text = [df[field].to_list() for field in fields]
    text = [f' {cfg.field_sep_token} '.join([f for f in t if f != ''])
            for t in zip(*text)]
    return text

def encode_in_chunks(text: List[str], tokenizer: PreTrainedTokenizerFast,
                     max_seq_length: int = 512, chunk_size: int = 16384
                     ) -> BatchEncoding[np.ndarray]:
    """Encode text in chunks to avoid OOM errors."""
    encodings = BatchEncoding()
    for offset in range(0, len(text), chunk_size):
        encodings_chunk = tokenizer(text[offset:offset+chunk_size],
            padding='max_length', truncation=True, max_length=max_seq_length,
            return_tensors='np')
        for k, v in encodings_chunk.items():
            if k not in encodings:
                encodings[k] = v
            else:
                encodings[k] = np.concatenate((encodings[k], v), axis=0)
    return encodings

def memmap_encodings(encodings: BatchEncoding[np.ndarray],
                     path: Union[str, Path]) -> np.memmap:
    """Save encodings to a memmap file."""
    values = np.array(list(encodings.values()))
    encodings = np.memmap(path, mode='w+', shape=values.shape, dtype=np.int16)
    encodings[:] = values[:]
    encodings.flush()
    encodings.setflags(write=False)
    return encodings
        
@timeit
def prepare_data(train: bool = True, load_enc: bool = False) -> None:
    """Prepare data for training or inference."""
    global cfg, logger, topics_df, topic_encodings, content_df,\
        content_encodings, correlations_df, sample_submission_df,\
        tc_edge_index, tt_edge_index, neighbor_sampler, tc_graph
    
    # Load raw data
    logger.info('Loading raw data.')
    topics_df = pd.read_csv(cfg.input_dir/'topics.csv')\
        .rename(columns={'id': 'topic_id'})\
        .set_index('topic_id')\
        .fillna({'title': '', 'description': ''})
    topics_df['level'] = topics_df.level.apply(lambda x: f'Level {x}')
    topics_df['num'] = range(len(topics_df))
    content_df = pd.read_csv(cfg.input_dir/'content.csv')\
        .rename(columns={'id': 'content_id'})\
        .set_index('content_id')\
        .fillna('')
    content_df['num'] = range(len(content_df))
    sample_submission_df = pd.read_csv(cfg.input_dir/'sample_submission.csv')
    
    if train:
        # Load training labels
        correlations_df = pd.read_csv(cfg.input_dir/'correlations.csv') \
            .set_index('topic_id')
        correlations_df['content_ids'] = correlations_df.content_ids.str \
            .split(' ')
    
        if cfg.max_topics: 
            # Optional topic subselction
            topics_df = topics_df.sample(cfg.max_topics) #TODO: sample by channel to make compatible with building tt-graph (need all parents)
            correlations_df = correlations_df.loc[
                topics_df.loc[topics_df.has_content].index]
            content_ids = correlations_df.explode('content_ids').content_ids \
                .unique()
            content_df = content_df.loc[content_ids]
            
        logger.info('Making CV splits.')
        # Get non-source topics for validation
        kfold_topics = topics_df.loc[topics_df.category != 'source']
        if cfg.val_size:
            # Subselect topics for validation
            kfold_topics = kfold_topics.sample(cfg.k_folds * cfg.val_size)
        kfold_topics = kfold_topics.index
        # Set aside topics for training only
        topics_df.loc[~topics_df.index.isin(kfold_topics), 'fold'] \
            = cfg.train_only_fold
        group_kfold = GroupKFold(cfg.k_folds)
        # Make CV splits grouped by channel
        for fold, (_, val_inds) in enumerate(group_kfold.split(
            X=topics_df.loc[kfold_topics],
            groups=topics_df.loc[kfold_topics, 'channel']
        )):
            topics_df.loc[kfold_topics[val_inds], 'fold'] = fold
        del group_kfold
        gc.collect()
           
    if load_enc:
        # Load tokenized text from existing memmap bin
        logger.info(f'Loading encodings.')
        topic_encodings = np.memmap(cfg.encodings_dir/'topic_enc.bin',
                                    mode='r', shape=(3, 76972, 256),
                                    dtype=np.int16)[..., :cfg.max_seq_length]
        content_encodings = np.memmap(cfg.encodings_dir/'content_enc.bin',
                                      mode='r', shape=(3, 154047, 256),
                                      dtype=np.int16)[..., :cfg.max_seq_length]
    else:
        # Or tokenize text and memmap to disk
        logger.info(f'Encoding topics ({len(topics_df)}).')
        text = make_repr(topics_df,
                         use_title=cfg.use_topic_title,
                         use_descr=cfg.use_topic_descr,
                         use_level=cfg.use_topic_level)
        encodings = encode_in_chunks(text, cfg.tokenizer, cfg.max_seq_length)
        topic_encodings = memmap_encodings(encodings,
                                           cfg.encodings_dir/'topic_enc.bin')
        
        logger.info(f'Encoding contents ({len(content_df)}).')
        text = make_repr(content_df,
                         use_title=cfg.use_content_title,
                         use_descr=cfg.use_content_descr,
                         use_text=cfg.use_content_text)
        encodings = encode_in_chunks(text, cfg.tokenizer, cfg.max_seq_length)
        content_encodings = memmap_encodings(encodings,
                                             cfg.encodings_dir/'content_enc.bin')
            
    if train:
        # Keep training columns
        topics_df = topics_df.loc[:, ['num', 'fold', 'parent', 'language',
                                      'has_content', 'title']]
        content_df = content_df.loc[:, ['num', 'kind', 'language', 'title']]
    
        # Build edge index for topic-content graph
        tc_pairs = correlations_df.explode('content_ids')
        source_nodes = id_to_int(tc_pairs.index, item_type='topic')
        target_nodes = id_to_int(tc_pairs.content_ids, item_type='content')
        tc_edge_index = [source_nodes, target_nodes]
        tc_edge_index = tensor(np.vstack(tc_edge_index), dtype=torch.int)
        _, sorted_idx = tc_edge_index[0].sort()
        tc_edge_index = tc_edge_index.gather(1, sorted_idx \
                                                .expand(tc_edge_index.size()))

        tc_graph = HeteroData()
        tc_graph['topic'].x = torch.tensor(topics_df.num.values, 
                                           dtype=torch.long).view(-1, 1)
        tc_graph['content'].x = torch.tensor(content_df.num.values,
                                             dtype=torch.long).view(-1, 1)
        tc_graph['topic', 'content'].edge_index = tc_edge_index.type(torch.long)
    else:
        # Keep inference columns
        topics_df = topics_df.loc[:, ['num', 'parent', 'language', 'title']]
        content_df = content_df.loc[:, ['num', 'kind', 'language', 'title']]
    
    # Build edge index and neighbor sampler for topic-topic graph 
    # (CV split by 'channel', i.e. disjoint train and val tt-graphs
    #     => global tt-graph)
    topic_parents = topics_df.loc[~topics_df['parent'].isna(), 'parent']
    tt_edge_index = [id_to_int(topic_parents.index),
                     id_to_int(topic_parents.values)]
    tt_edge_index = tensor(np.vstack(tt_edge_index), dtype=torch.int)
    _, sorted_idx = tt_edge_index[0].sort()
    tt_edge_index = tt_edge_index.gather(1, sorted_idx.expand(
                                                tt_edge_index.size()))
    
    undirected_tt_edge_index = torch.cat((tt_edge_index, tt_edge_index.flip(0)),
                                         1).type(torch.long)
    neighbor_sampler = NeighborSampler(edge_index=undirected_tt_edge_index,
                                       sizes=cfg.neighborhood_sizes,
                                       shuffle=True,
                                       return_e_id=False)
    
    logger.info(f"Successfully loaded and processed data.")
    
# --------------------------------------------------------------------------- #
#                                  Model                                      #
# --------------------------------------------------------------------------- #
class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
    
    @property
    def mode(self) -> str:
        return 'train' if self.training else 'val'
    
    def count_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

class MeanPooling(BaseModule):
    def __init__(self):
        super().__init__()
    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) \
        -> Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1) \
                                            .expand(token_embeddings.size()) \
                                            .float()
        return (torch.sum(token_embeddings * input_mask_expanded, 1) /
                torch.clamp(input_mask_expanded.sum(1), min=cfg.eps))

class SentenceEncoder(BaseModule):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(cfg.encoder_backbone)
        if cfg.gradient_checkpointing:
            config.gradient_checkpointing = True
        backbone = AutoModel.from_pretrained(cfg.encoder_backbone,
                                             config=config)
        backbone.resize_token_embeddings(len(cfg.tokenizer))
        self.config = config
        self.backbone = backbone
        self.pool = MeanPooling()
    def forward(self, encodings: BatchEncoding[Tensor]) -> Tensor:
        embeddings = self.backbone(**encodings)
        embeddings = self.pool(embeddings.last_hidden_state,
                               encodings.attention_mask)
        return embeddings

class SAGE(BaseModule):
    def __init__(self, in_out_channels: int, hidden_channels: int,
                 num_layers: int = 3, aggr: str = 'mean') -> None:
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_out_channels, hidden_channels, aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr))
        self.convs.append(SAGEConv(hidden_channels, in_out_channels, aggr))
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x: Tensor, adjs: List[Tuple]) -> Tensor:
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x
    
class TopicEncoder(BaseModule):
    def __init__(self):
        super().__init__()
        self.sentence_encoder = SentenceEncoder()
        D = self.sentence_encoder.config.hidden_size
        self.sage_conv = SAGE(D,
                              hidden_channels=D,
                              num_layers=len(cfg.neighborhood_sizes),
                              aggr='mean')
    def forward(self, encodings: BatchEncoding[Tensor], adjs: List[Tuple]) \
        -> Dict[Tensor]:
        embeddings = self.sentence_encoder(encodings)
        if not cfg.bypass_post:
            embeddings = self.sage_conv(embeddings, adjs)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return dict(topic_emb=embeddings)
    
class ContentEncoder(BaseModule):
    def __init__(self):
        super().__init__()
        self.sentence_encoder = SentenceEncoder()
        D = self.sentence_encoder.config.hidden_size
        dense = nn.Linear(D, D, bias=True)
        nn.init.eye_(dense.weight)
        nn.init.zeros_(dense.bias)
        self.dense = dense
    def forward(self, encodings: BatchEncoding[Tensor]) -> Dict[Tensor]:
        embeddings = self.sentence_encoder(encodings)
        if not cfg.bypass_post:
            embeddings = self.dense(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return dict(content_emb=embeddings)

class BiEncoder(BaseModule):
    def __init__(self):
        super().__init__()
        self.topic_encoder = TopicEncoder()
        self.content_encoder = ContentEncoder()
    def forward(self, topic_enc: BatchEncoding[Tensor], adjs: List[Tuple],
                content_enc: BatchEncoding[Tensor]) -> Dict[Tensor]:
        topic_emb = self.topic_encoder(topic_enc, adjs)['topic_emb']
        content_emb = self.content_encoder(content_enc)['content_emb']
        return dict(topic_emb=topic_emb, content_emb=content_emb)
    def freeze_backbone(self):
        for param in self.topic_encoder.sentence_encoder.parameters():
            param.requires_grad = False
        for param in self.content_encoder.sentence_encoder.parameters():
            param.requires_grad = False
    
class CrossEncoderClassifier(BaseModule):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(cfg.encoder_backbone)
        config.num_labels = 1
        if cfg.gradient_checkpointing: config.gradient_checkpointing = True
        backbone = AutoModelForSequenceClassification \
                        .from_pretrained(cfg.encoder_backbone, config=config)
        backbone.resize_token_embeddings(len(cfg.tokenizer))
        self.config = config
        self.backbone = backbone
    def forward(self, cross_enc: BatchEncoding[Tensor]) -> Dict[Tensor]:
        logits = self.backbone(**cross_enc).logits
        return dict(logits=logits.view(-1))
    
@torch.no_grad()
def infer_pred(loader: DataLoader, model: nn.Module) -> Dict[str, np.ndarray]:
    """Forward pass a model on a given loader and return predictions."""
    N = len(loader.dataset)
    n_batches, loader = len(loader), iter(loader)
    pred = defaultdict(list)
    model.eval()
    for i in tqdm(range(n_batches), desc=f"Inferring for {str(N):<8}"):
        batch = next(loader)
        with torch.autocast(device_type=cfg.device, dtype=cfg.cast_dtype):
            outputs = model(*batch['inputs'])
        for k, v in outputs.items():
            pred[k].append(v)
    for k, v in pred.items():
        pred[k] = torch.cat(v).cpu().numpy()
    return pred

# --------------------------------------------------------------------------- #
#                                   Data                                      #
# --------------------------------------------------------------------------- #
class RetrieverTrainingSet(Dataset):
    def __init__(self, topic_index=None):
        self.topic_index = topic_index if (topic_index is not None) \
            else topics_df.num.values
    def __len__(self):
        return len(self.topic_index)
    def __getitem__(self, item_num):
        return self.topic_index[item_num]

class RerankerTrainingSet(Dataset):
    def __init__(self, pairs, labels):
        assert pairs is not None, "Cannot create reranker train set \
            without an index of topic-content pairs."
        self.pairs = pairs
        self.labels = labels
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, item_num):
        return (self.pairs[item_num], self.labels[item_num])

class RetrieverTestSet(Dataset):
    def __init__(self, items):
        assert items is not None, "Cannot create retriever test set \
            without a sequence of topics or contents."
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, item_num):
        return self.items[item_num]

class RerankerTestSet(Dataset):
    def __init__(self, pairs):
        assert pairs is not None, "Cannot create reranker test set \
            without a sequence of topic-content pairs."
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, item_num):
        return self.pairs[item_num]

def get_pairs(content_matrix: Union[Tensor, np.ndarray],
              topic_index: Union[Tensor, np.ndarray, None] = None) -> Tensor:
    """Returns a tensor of pairs of topic and content numbers, where each
    element of 'topic_index' is paired to all elements of the corresponding row
    in 'content_matrix'."""
    topic_index = topic_index if (topic_index is not None) \
        else topics_df.num.values
    if not isinstance(topic_index, Tensor):
        topic_index = tensor(topic_index)
    if not isinstance(content_matrix, Tensor):
        content_matrix = tensor(content_matrix)
    topic_index = topic_index.type(torch.int32)
    content_matrix = content_matrix.type(torch.int32)
    pairs = torch.cat((topic_index.view(-1, 1, 1) \
                                  .expand(-1, content_matrix.size(1), 1),
                       content_matrix.unsqueeze(-1)), dim=-1)
    pairs = pairs.reshape(content_matrix.numel(), 2)
    return pairs
    
def label_pairs(pairs: Tensor) -> Tensor:
    """Classify each pair of topic and content numbers in 'pairs' using the GT
    topic-content graph."""
    # Sort pairs by topic number for efficient search
    _, sorted_idx = pairs.T[0].sort()
    pairs = pairs[sorted_idx]
    # Label pairs in chunks to avoid OOM
    def label_chunk(offset=0, chunk_size=8192):
        # Search for GT labels in subspace of edges containing query topics
        sub_edge_index = get_subgraph_edge_index(
            pairs[offset:offset+chunk_size].T[0].unique(), tc_edge_index)
        return (pairs[offset:offset+chunk_size].unsqueeze(1)
                == sub_edge_index.T).all(-1).any(1).type(torch.int8)
    labels = label_chunk()
    while labels.size(0) < pairs.size(0):
        labels = torch.cat((labels, label_chunk(offset=labels.size(0))), axis=0)
    # Return labels to original order
    unsort_labels = torch.zeros(labels.size())
    unsort_labels[sorted_idx] = labels.type(torch.float32)
    return unsort_labels
    
def get_enc(item_nums: Sequence[int], item_type: str = 'topic',
            out_type: str = 'tensor') -> BatchEncoding[Tensor] | Tensor:
    """Get encodings for the slice of 'item_nums' of type 'item_type'."""
    if item_type == 'topic':
        enc_vals = tensor(topic_encodings[:, item_nums, :], dtype=torch.int)
    elif item_type == 'content':  
        enc_vals = tensor(content_encodings[:, item_nums, :], dtype=torch.int)
    else:
        raise Exception(f'item_type: {item_type} not implemented')
    if out_type == 'dict':
        enc_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        return BatchEncoding(data=dict(zip(enc_keys, enc_vals)))
    return enc_vals

def get_cross_enc(topic_nums: Sequence[int], content_nums: Sequence[int],
                  out_type: str = 'tensor') -> BatchEncoding[Tensor] | Tensor:
    """Get encodings for the crossed slices of 'topic_nums' and 'content_nums',
    paired by input order."""
    assert len(content_nums) == len(topic_nums), 'No. topics and no. contents \
        must match for cross encoding of topic-content pairs'
    t_enc = get_enc(topic_nums, 'topic', out_type='tensor')
    c_enc = get_enc(content_nums, 'content', out_type='tensor')
    c_enc[1] = c_enc[2] # token type 1 for content
    cross_enc = torch.cat((t_enc, c_enc), dim=2)
    _, indices = cross_enc[2].sort(dim=1, descending=True, stable=True)
    cross_enc = cross_enc.gather(2, indices.expand(cross_enc.size()))
    if out_type == 'dict':
        enc_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        return BatchEncoding(data=dict(zip(enc_keys, cross_enc)))
    return cross_enc
    
def get_subgraph_edge_index(sub_nodes: Sequence[int], edge_index: Tensor, 
                            node_type: str = 'source',
                            return_mask: bool = False) -> Tensor:
    """Returns the subgraph of 'edge_index' containing only edges with nodes
    of type 'node_type' in 'sub_nodes'."""
    if node_type == 'source':
        node_type = 0
    elif node_type == 'target':
        node_type = 1
    else:
        raise Exception(f'node_type: {node_type} not implemented.')
    if not isinstance(sub_nodes, Tensor):
        sub_nodes = tensor(sub_nodes)
    idx_mask = (((edge_index[node_type].unsqueeze(-1) - sub_nodes) == 0) \
        .sum(-1) == 1)
    if return_mask:
        return edge_index[:, idx_mask], idx_mask
    return edge_index[:, idx_mask]

def sample_edges(edge_index: Tensor) -> Tensor:
    """Sample one random edge per distinct source node in 'edge_index'."""
    idx = torch.randperm(edge_index[0].nelement())
    edge_perm = edge_index[:, idx]
    unique, inv_idx, counts = edge_perm[0].unique(sorted=True,
                                    return_inverse=True, return_counts=True)
    _, ind_sorted = inv_idx.sort(stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((tensor([0]), cum_sum[:-1]))
    return edge_perm[:, ind_sorted[cum_sum]]

def prepare_retriever_batch(topic_nums: Sequence[int]) -> Dict:
    """Return a training input for the retriever given a batch of topic
    numbers."""
    if not isinstance(topic_nums, Tensor):
        topic_nums = tensor(topic_nums, dtype=torch.long)
    else:
        topic_nums = topic_nums.type(torch.long)

    # Sample one content per topic with content
    sub_edge_index = tc_graph.subgraph(
        {'topic': topic_nums, 'content': tc_graph['content'].x.squeeze()}) \
        ['topic', 'content'].edge_index
    _, content_nums = sample_edges(sub_edge_index).type(torch.long)

    # Catch off-diagonal correlations (content shared between topics in batch)
    sub_edge_index = tc_graph.subgraph({'topic': topic_nums,
                                        'content': content_nums})\
                                            ['topic', 'content'].edge_index

    # Sort edge_index by input order of topics
    _, sorted_topic_idx = topic_nums.sort()
    topic_idx = sorted_topic_idx[sub_edge_index[0]]
    sub_edge_index[0] = topic_idx
    content_nums = content_nums.unique(sorted=True)
    n_topics, n_contents = topic_nums.size(0), content_nums.size(0)

    # Make in-batch positive and negative labels from edge_index
    labels = torch.sparse_coo_tensor(sub_edge_index,
                                     torch.ones(sub_edge_index.size(1)),
                                     (n_topics, n_contents)).to_dense()
    
    # Get topic and content encodings
    content_enc = get_enc(content_nums, item_type='content', out_type='dict')
    if cfg.bypass_post:
        topic_enc = get_enc(topic_nums, item_type='topic', out_type='dict') 
        adjs = []
    else: 
        # If forward pass involves graph conv, sample topic neighbors and get 
        # additional encodings
        _, n_id, adjs = neighbor_sampler.sample(topic_nums.type(torch.long))
        # n_id: (first) input topic_nums
        # + (then) neighbors at all depths required for graph conv
        topic_enc = get_enc(n_id, item_type='topic', out_type='dict')
        adjs = [adj.to(cfg.device) for adj in adjs]
    
    # Move to device
    topic_enc.data = tensor_dict_to(topic_enc,
                                    pin_memory=(cfg.device == 'cuda'),
                                    device=cfg.device)
    content_enc.data = tensor_dict_to(content_enc,
                                      pin_memory=(cfg.device == 'cuda'),
                                      device=cfg.device)     
    if cfg.device == 'cuda':
        labels = labels.to_dense().pin_memory().to('cuda', non_blocking=True)
    else:
        labels = labels.to(cfg.device)
        
    return dict(inputs=(topic_enc, adjs, content_enc), labels=labels)

def prepare_reranker_batch(tuples: Sequence[Tuple[Tuple[int, int], int]]) \
                           -> Dict:
    """Return a training input for the reranker given a batch of tuples of the
    form ([t_num, c_num], label)."""
    pairs, labels = zip(*tuples)
    pairs = torch.stack(pairs)
    cross_enc = get_cross_enc(*pairs.T, out_type='dict')
    cross_enc.data = tensor_dict_to(cross_enc,
                                    pin_memory=(cfg.device == 'cuda'),
                                    device=cfg.device)
    labels = torch.stack(labels).type(torch.float32)
    if cfg.device == 'cuda':
        labels = labels.pin_memory().to(cfg.device, non_blocking=True)
    else:
        labels = labels.to(cfg.device)
    return dict(inputs=(cross_enc,), labels=labels)

def prepare_retriever_test_topic_batch(topic_nums: Sequence[int]) -> Dict:
    """Return a test input topic batch for the retriever."""
    if not isinstance(topic_nums, Tensor):
        topic_nums = tensor(topic_nums, dtype=torch.int32)
        
    if cfg.bypass_post: # For forward pass with bypassed graph conv
        topic_enc = get_enc(topic_nums, item_type='topic', out_type='dict')
        adjs = []
    else:
        _, n_id, adjs = neighbor_sampler.sample(topic_nums.type(torch.long))
        topic_enc = get_enc(n_id, item_type='topic', out_type='dict')
        adjs = [adj.to(cfg.device) for adj in adjs]
        
    topic_enc.data = tensor_dict_to(topic_enc,
                                    pin_memory=(cfg.device == 'cuda'),
                                    device=cfg.device)
    return dict(inputs=(topic_enc, adjs))

def prepare_retriever_test_content_batch(content_nums: Sequence[int]) -> Dict:
    """Return a test input content batch for the retriever."""
    if not isinstance(content_nums, Tensor):
        content_nums = tensor(content_nums, dtype=torch.int32)
    content_enc = get_enc(content_nums, item_type='content', out_type='dict')
    content_enc.data = tensor_dict_to(content_enc,
                                      pin_memory=(cfg.device == 'cuda'),
                                      device=cfg.device)
    return dict(inputs=(content_enc,))

def prepare_reranker_test_batch(pairs: Sequence[Sequence[int, int]]) -> Dict:
    """Return a test input batch for the reranker."""
    pairs = torch.stack(pairs)
    cross_enc = get_cross_enc(*pairs.T, out_type='dict')
    cross_enc.data = tensor_dict_to(cross_enc,
                                    pin_memory=(cfg.device == 'cuda'),
                                    device=cfg.device)
    return dict(inputs=(cross_enc,))

# --------------------------------------------------------------------------- #
#                               Post-processing                               #
# --------------------------------------------------------------------------- #
class Retriever(BiEncoder):
    def __init__(self):
        super().__init__()
        self.name = 'retriever'
        self.stage_num = 1
        if cfg.freeze_backbone:
            self.freeze_backbone()
            
    def make_recos(self, loaders: Tuple[DataLoader, DataLoader]) -> Tensor:
        """Retrieve topic-content pair recommendations with biencoder in test
        mode, given topic and content data loaders."""
        topic_loader, content_loader = loaders
        n_samples = ' x '.join([f'{len(l.dataset)}' for l in loaders])
        logger.info(f'Inferring {self.stage_name} predictions for \
                    {n_samples} samples.')
        
        # Predict embeddings
        topic_emb = infer_pred(topic_loader,
                               self.topic_encoder)['topic_emb']
        content_emb = infer_pred(content_loader,
                                 self.content_encoder)['content_emb']
        
        # Find top-k similar contents for each topic
        top_k_sim_val, top_k_sim_idx = retrieve_top_k_contents(topic_emb,
                                                               content_emb)
        
        # Filter out pairs with similarity below a dynamic threshold
        dynamic_thresh = (1 - cfg.sim_margin) * top_k_sim_val[:, 0][:, None]
        mask = (top_k_sim_val >= dynamic_thresh).flatten()
        stage_1_pairs = get_pairs(top_k_sim_idx,
                                  topic_loader.dataset.item_index)[mask]
        
        return stage_1_pairs

class Reranker(CrossEncoderClassifier):
    def __init__(self):
        super().__init__()
        self.name = 'reranker'
        self.stage_num = 2
    
    def make_recos(self, loaders: Tuple[DataLoader]) -> Tensor:
        """Rerank/filter topic-content pair recommendations with cross-encoder
        in test mode, given a data loader of topic-content pairs."""
        pair_loader = loaders[0]
        logger.info(f'Inferring {self.stage_name} predictions for \
                    {len(pair_loader.dataset)} samples.')
        
        logits = infer_pred(pair_loader, self)['logits']
        reco_mask = rerank_top_k_contents(logits)
        stage_2_pairs = pair_loader.dataset.pairs[reco_mask]
        return stage_2_pairs
    
def get_model(stage):
    if stage == 1:
        return Retriever().to(cfg.device)
    elif stage == 2:
        return Reranker().to(cfg.device)
    else:
        raise Exception(f'Stage {stage} not implemented.')
    
def index_corpus(corpus_vectors: np.ndarray) -> faiss.IndexIVF:
    """Return a searchable IVF Faiss index that is quantized using HNSW and
    trained on the given corpus vectors."""
    d = corpus_vectors.shape[-1]
    quantizer = faiss.IndexHNSWFlat(d, cfg.index_nlinks,
                                    faiss.METRIC_INNER_PRODUCT)
    quantizer.hnsw.efConstruction = cfg.index_efConstruction
    quantizer.hnsw.efSearch = cfg.index_efSearch
    index = faiss.IndexIVFFlat(quantizer, d, cfg.index_nclusters,
                                faiss.METRIC_INNER_PRODUCT)
    index.train(corpus_vectors)
    index.add(corpus_vectors)
    index.nprobe = cfg.index_nprobe
    return index

def top_k_cos_sim_in_chunks(query_vectors: Tensor, corpus_vectors: Tensor,
                            k: int = 100, chunk_size: int = 512) \
                            -> Tuple[np.ndarray, np.ndarray]:
    """Return the top k cosine similarities and indices between query and
    corpus vectors (evaluated in chunks to avoid OOM errors)."""
    a, b, s = query_vectors, corpus_vectors, chunk_size
    if not isinstance(a, Tensor):
        a = tensor(a, dtype=torch.float, device=cfg.device)
    if not isinstance(b, Tensor):
        b = tensor(b, dtype=torch.float, device=cfg.device)
    top_k_out_val = torch.zeros((len(a), k), dtype=torch.float,
                                device=a.device)
    top_k_out_idx = torch.zeros((len(a), k), dtype=torch.int,
                                device=a.device)
    
    for i, a_chunk in enumerate(a.split(s)):
        sim_chunk = torch.zeros((len(a_chunk), len(b)), dtype=torch.float,
                                device=a.device)
        for j, b_chunk in enumerate(b.split(s)):
            sim_chunk[:, j*s:(j+1)*s] = a_chunk @ b_chunk.T
        top_k_chunk_val, top_k_chunk_idx = torch.topk(sim_chunk, k)
        top_k_out_val[i*s:(i+1)*s, :] = top_k_chunk_val
        top_k_out_idx[i*s:(i+1)*s, :] = top_k_chunk_idx
        
    return top_k_out_val.cpu().numpy(), top_k_out_idx.cpu().numpy()

def retrieve_top_k_contents(topic_emb: Union[Tensor, np.ndarray],
                            content_emb: Union[Tensor, np.ndarray]) \
                            -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the top-k content indices (and similarity values) for each
    topic given topic and content embeddings."""    
    top_k_sim_val = np.zeros((len(topic_emb), cfg.top_k), dtype=float)
    top_k_sim_idx = np.zeros((len(topic_emb), cfg.top_k), dtype=np.int32)
    languages = topics_df.language.unique()
    
    # Separate search by language
    logger.info(f'Retrieving top-{cfg.top_k} contents.')
    for lang in languages:
        topic_nums = topics_df.loc[topics_df.language==lang, 'num'].values
        content_nums = content_df.loc[content_df.language==lang, 'num'].values
        if len(content_nums) == 0:
            # Default to enlglish content if none in given language
            content_nums = content_df.loc[content_df.language=='en', 'num'] \
                                     .values
        t_emb_sub = topic_emb[topic_nums]
        c_emb_sub = content_emb[content_nums]
        # Use approximate index if corpus is too large for exact search
        if len(content_nums) > cfg.use_index_above:          
            index = index_corpus(c_emb_sub)
            top_k_val, top_k_idx = index.search(t_emb_sub, cfg.top_k)
        else:
            top_k_val, top_k_idx = top_k_cos_sim_in_chunks(t_emb_sub, c_emb_sub,
                                                           cfg.top_k)
        top_k_sim_val[topic_nums] = top_k_val
        top_k_sim_idx[topic_nums] = content_nums[top_k_idx]

    return top_k_sim_val, top_k_sim_idx

def rerank_top_k_contents(logits: Union[Tensor, np.ndarray]) \
                          -> np.ndarray[bool]:
    """Return a mask retaining indices where logits are above a threshold."""
    if not isinstance(logits, Tensor):
        logits = tensor(logits, dtype=torch.float)
    reco_mask = (nn.Sigmoid()(logits) > cfg.rerank_threshold).cpu().numpy() #TODO: make dynamic thresh
    return reco_mask

def reco_pairs_to_series(pairs: Union[Tensor, np.ndarray],
                         all_topics: bool = False) -> pd.Series:
    """Convert a tensor of topic-content pairs to a series of content id lists
    indexed by topic id."""
    topic_ids = int_to_id(pairs[:, 0], 'topic')
    content_ids = int_to_id(pairs[:, 1], 'content')
    reco_series = pd.Series(content_ids, index=topic_ids, name='content_ids') \
                    .groupby(level=0).agg(lambda x: [idx for idx in x])  
    if all_topics:
        topics_wo = topics_df.index[~topics_df.index.isin(topic_ids)].values
        wo_recos = pd.DataFrame({'topic_id': topics_wo, 'content_ids': ''}) \
                     .set_index('topic_id').squeeze().apply(lambda x: [])
        reco_series = pd.concat([reco_series, wo_recos], axis=0)
    return reco_series

def evaluate_recommendations(recommendations: pd.Series) -> Dict[float]:
    """Evaluate content recommendations by computing recall, precision and
    F2-score for each topic and averaging by topic."""
    topic_ids = recommendations.index
    
    # true positives = relevant /\ recommended
    recommended_and_relevant = [
        set(Topic(t).content_ids).intersection(set(recommendations.loc[t])) 
        for t in topic_ids
    ]
    recommended_and_relevant = pd.Series(recommended_and_relevant,
                                         index=topic_ids)
    
    # recall = |relevant /\ recommended| / |relevant|
    recall = np.array([
        ((float(len(recommended_and_relevant.loc[t])) + cfg.eps) / 
         (len(Topic(t).content_ids) + cfg.eps))
        for t in topic_ids
    ])
    
    # precision = |relevant /\ recommended| / |recommended|
    precision = np.array([
        (float(len(recommended_and_relevant.loc[t])) /
         (len(recommendations.loc[t]) + cfg.eps))
        for t in topic_ids
    ])
    
    # fBscore = (1 + B^2) * precision * recall / (B^2 * precision + recall)
    f2score = 5 * precision * recall / (4 * precision + recall)
    
    return dict(recall=round(recall.mean(), cfg.precision),
                precision=round(precision.mean(), cfg.precision),
                f2score=round(f2score.mean(), cfg.precision))
    
# --------------------------------------------------------------------------- #
#                                 Trainer                                     #
# --------------------------------------------------------------------------- #
def similarity_nll(topic_emb: Tensor, content_emb: Tensor,
                   labels: Tensor[int]) -> Tensor:
    """Compute the negative log-likelihood of the dot-product similarity
    between topic and content embeddings."""
    assert topic_emb.size()[1] == content_emb.size()[1], "Topic and content \
        embedding dimensionalities do not match"
    sim = topic_emb @ content_emb.T / cfg.temperature
    return nn.CrossEntropyLoss()(sim, labels)

def logits_bce(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute the binary cross-entropy loss given logits and labels."""
    return nn.BCEWithLogitsLoss(pos_weight=tensor(2))(logits, labels)

class StagedTrainer:
    def __init__(self, fold: Optional[int] = None) -> None:
        self.fold = fold
        self.epoch = 0
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.history = dict(retriever=defaultdict(list),
                            reranker=defaultdict(list))
        self.history_idx_offset = 0
        self.recos = defaultdict(None)
        self.best_model = defaultdict(lambda: float('inf'))
        self.log_precision = cfg.precision
        self.stage_epochs = {1: cfg.retriever_epochs, 
                             2: cfg.reranker_epochs}
        self.loader_args = dict(
            shuffle=cfg.shuffle,
            generator=g['pt'],
            # num_workers=4,
        )
        self.tracked = 'val_loss'
        self.keep_best_only = True
        
        # Only use topics with content (avoid diverging NLL)
        self.train_ids = topics_df.loc[(topics_df.has_content==True)
                                       & (topics_df.fold != fold)].num.values
        self.val_ids = topics_df.loc[(topics_df.has_content==True)
                                     & (topics_df.fold == fold)].num.values 
        
    def setup_stage(self, stage: int, from_ckpt: Optional[bool] = None) -> None:
        """Setup data, model and loggers for a given stage of training."""
        if stage == 1:
            self.train_set = RetrieverTrainingSet(self.train_ids)
            self.val_set = RetrieverTrainingSet(self.val_ids)
            self.loader_args.update(batch_size=cfg.retriever_batch_size,
                                    collate_fn=prepare_retriever_batch)
            self.loss_fn = similarity_nll
            self.stage_name = 'retriever'
        elif stage == 2:
            # Get all ground truth tc-edges and add false edges, i.e negative 
            # samples, randomly
            size = (len(topics_df), len(content_df))
            edge_index, _ = add_random_edge(tc_edge_index.type(torch.long),
                                            num_nodes=size,
                                            p=cfg.reranker_neg_sample_ratio)
            labels = torch.zeros(edge_index.size(1), dtype=torch.float32)
            labels[:tc_edge_index.size(1)] = 1.
            _, _, train_mask = bipartite_subgraph(
                (tensor(self.train_ids), tensor(content_df.num.values)),
                edge_index, size=size, return_edge_mask=True)
            train_pairs, = edge_index.T[train_mask]
            train_labels = labels[train_mask]
            val_pairs = edge_index.T[~train_mask]
            val_labels = labels[~train_mask]
            
            # Create train and val sets
            self.train_set = RerankerTrainingSet(train_pairs, train_labels)
            self.val_set = RerankerTrainingSet(val_pairs, val_labels)
            self.loader_args.update(batch_size=cfg.reranker_batch_size,
                                    collate_fn=prepare_reranker_batch)  
            self.loss_fn = logits_bce
            self.stage_name = 'reranker'
        
        self.stage_num = stage
        self.train_loader = DataLoader(self.train_set, **self.loader_args)
        self.val_loader = DataLoader(self.val_set, **self.loader_args)
        
        # Load model training history if using a checkpoint
        self.history_idx_offset = 0
        if from_ckpt is not None:
            checkpoint = torch.load(from_ckpt)
            [self.history[self.stage_name][m[5:]].append(checkpoint[m])
             for m in checkpoint if m.startswith('curr_')]
            self.history_idx_offset = 1
            self.update_best_model()
            # Override update method assignments
            self.best_model[f'{self.stage_name}_ckpt_path'] = from_ckpt
            self.best_model[f'{self.stage_name}_prev_ckpt_path'] = from_ckpt
            
            best_tracked = self.best_model[f'{self.stage_name}_{self.tracked}']
            best_tracked = round(best_tracked, self.log_precision)
            logger.info(f'Resuming from checkpoint: {from_ckpt} \
                ({self.tracked}: {best_tracked}).')
        
        # Logs
        self.tb_writer = SummaryWriter(self.tblg_path)
        logger.info(
            f'Stage {stage} ({self.stage_name}) setup complete. \
            Training size: {len(self.train_set)}. \
            Validation size: {len(self.val_set)}. \
            Batch size: {self.loader_args["batch_size"]}.')
  
    @property
    def tblg_path(self) -> Path:
        """Return path to tensorboard log file."""
        f = f'{self.stage_name}_trainer_fold{self.fold}\_{self.timestamp}'
        return cfg.tb_logs_dir/f
    
    def ckpt_path(self, stage: Optional[int] = None,
                  epoch: Optional[int] = None) -> Path:
        """Return path to checkpoint file for a given stage and epoch."""
        if stage is None:
            stage = self.stage_name
        if epoch is None:
            epoch = self.epoch
        return cfg.checkpoints_dir/f'{stage}_ckpt_{self.timestamp}_{epoch}.pt'
    
    def run_epoch(self, model: BaseModule) -> float:
        """Run a single epoch of training or validation on a model and return
        epoch's loss value."""
        # Initialize
        loader = self.train_loader if model.mode == 'train' else self.val_loader
        n_batches, loader = len(loader), iter(loader)
        batch = next(loader)
        total_loss = 0.
        tqdm_desc = f"Epoch {self.epoch} ({model.mode:<5})"
        
        for i in tqdm(range(n_batches), desc=tqdm_desc):
            # Forward pass
            with torch.autocast(device_type=cfg.device, dtype=cfg.cast_dtype):
                outputs = model(*batch['inputs'])
                loss = self.loss_fn(**outputs, labels=batch['labels'])
                total_loss += loss.detach()

            # Prefetch next batch
            if i + 1 < n_batches: batch = next(loader)

            if model.mode == 'train':
                # Backward pass
                self.scaler.scale(loss).backward()
                # Gradient accumulation
                if (not (i + 1) % cfg.grad_accumulation_steps) \
                    or (i + 1 == n_batches):
                    # Optimizer step
                    if cfg.max_grad_norm != 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    # Scheduler step when using batch-level schedule
                    self.scheduler.step()
            
        # Log
        epoch_loss = total_loss.item()/n_batches
        self.history[self.stage_name][f'{model.mode}_loss'].append(epoch_loss)
        return epoch_loss
    
    def metrics(self, split: Optional[str] = None) -> List[str]:
        """Get a list of names of logged metrics given a string splitting
        pattern."""
        all_metrics = list(self.history[self.stage_name].keys())
        if split is None:
            return all_metrics
        if split in ('train', 'val'):
            return [m for m in all_metrics if f'{split}_' in m]
        if split == 'name':
            return [m.split('_')[-1] for m in all_metrics]
        else:
            return [m.split(split) for m in all_metrics]
    
    def get_metric(self, metric: str) -> float:
        idx = self.epoch - 1 + self.history_idx_offset
        return self.history[self.stage_name].get(metric)[idx]
    
    def update_best_model(self) -> None:
        """Update variables tracking the best performing model so far."""
        prev_best = self.best_model[f'{self.stage_name}_{self.tracked}']
        curr_val = self.get_metric(self.tracked)
        if curr_val < prev_best:
            prev_ckpt = self.best_model[f'{self.stage_name}_ckpt_path']
            self.best_model[f'{self.stage_name}_{self.tracked}'] = curr_val
            self.best_model[f'{self.stage_name}_prev_ckpt_path'] = prev_ckpt
            self.best_model[f'{self.stage_name}_ckpt_path'] = self.ckpt_path()
    
    def save_checkpoint(self, model: BaseModule) -> None:
        best_tracked = self.best_model.get(f'{self.stage_name}_{self.tracked}')
        if self.keep_best_only:
            # Skip saving if current model is not the best performing
            if best_tracked != self.get_metric(self.tracked):
                return
            # Delete previous best checkpoint if it exists
            prev_ckpt = self.best_model[f'{self.stage_name}_prev_ckpt_path']
            if isinstance(prev_ckpt, Path) and prev_ckpt.is_file():
                os.remove(prev_ckpt)
                
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'fold': self.fold,
            'stage': self.stage_name,
            'epoch': self.epoch,
            f'best_{self.tracked}': best_tracked,
            **{f'curr_{m}': self.get_metric(m) for m in self.metrics()},
            'config': cfg,
        }
        torch.save(checkpoint, self.ckpt_path())
        logger.info(f'Checkpoint saved to: {self.ckpt_path()}')
        return
    
    def load_checkpoint(self, model: BaseModule, checkpoint: dict) -> None:
        model.load_state_dict(checkpoint['model'])
        model.to(cfg.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
    
    def write_tb_log(self) -> None:
        metric_logs = [
            (f"Epochs/{self.stage_name}_{m}",
             {'train': self.get_metric(f'train_{m}'),
              'val':   self.get_metric(f'val_{m}')}) 
                for m in self.metrics('name')]
        for named_dict in metric_logs:
            self.tb_writer.add_scalars(*named_dict, self.epoch)
        self.tb_writer.flush()
    
    def make_recos(self, model): #TODO: move to Retriver/Reranker classes and return pairs
        if self.stage_name == 'retriever':
            pairs = None
            topic_set = RetrieverTestSet(topics_df.num.values)
            content_set = RetrieverTestSet(content_df.num.values)
            topic_loader = DataLoader(topic_set,
                                shuffle=False,
                                batch_size=cfg.retriever_batch_size,
                                collate_fn=prepare_retriever_test_topic_batch)
            content_loader = DataLoader(content_set,
                                shuffle=False,
                                batch_size=cfg.retriever_batch_size,
                                collate_fn=prepare_retriever_test_content_batch)
            loaders = (topic_loader, content_loader)
        elif self.stage_name == 'reranker':
            pairs = self.recos['stage_1_pairs']
            pair_set = RerankerTestSet(pairs)
            pair_loader = DataLoader(pair_set,
                                shuffle=False,
                                batch_size=cfg.reranker_batch_size,
                                collate_fn=prepare_reranker_test_batch)
            loaders = (pair_loader,)
        
        reco_pairs = model.make_recos(loaders)
        self.recos[f'stage_{self.stage_num}_pairs'] = reco_pairs
        
    def eval_recos(self): #TODO: Input pairs with option for OOF or full test
        # Evaluate recos
        reco_pairs = self.recos[f'stage_{self.stage_num}_pairs']
        recos = reco_pairs_to_series(reco_pairs)
        logger.info(f'Evaluating predictions.')
        eval_metrics = evaluate_recommendations(recos) 
        
        # Log
        to_logger = []
        for m, v in eval_metrics.items():
            self.best_model[f'{self.stage_name}_{m}'] = v
            to_logger.append(f'{m}: {round(v, self.log_precision)}.')
        logger.info(' '.join(to_logger))
    
    def train(self, model: BaseModule) -> None:
        assert self.stage_num in self.stage_epochs, 'Check stage setup.'
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          cfg.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp
                                                        & (cfg.device=="cuda")))
        steps_per_epoch = int(np.ceil(len(self.train_loader)
                                      * 1./cfg.grad_accumulation_steps))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
            max_lr = cfg.learning_rate,
            steps_per_epoch = steps_per_epoch,
            epochs = self.stage_epochs[self.stage_num],
            div_factor = 25, #TODO: make cfg parameters
            pct_start = 0.3,
            final_div_factor = 100,
        )
        
        logger.info(f'Begin training of {model.count_params()} parameters.\n\
            Fold: {self.fold}. Top-k: {cfg.top_k}. \
            Gradient checkpointing: {cfg.gradient_checkpointing}. \
            Automatic mixed precision: {cfg.use_amp}. \
            GPU available: {cfg.device=="cuda"}.')       
        # Epoch loop
        for epoch in np.arange(1, 1 + self.stage_epochs[self.stage_num]):
            self.epoch = epoch
            # Train
            model.model.train(True)
            train_loss = self.run_epoch(model)
            # Validate
            model.model.train(False)
            with torch.no_grad():
                val_loss = self.run_epoch(model)
            logger.info(f'Epoch {self.epoch} - \
                train loss: {round(train_loss, self.log_precision)}, \
                val loss: {round(val_loss, self.log_precision)}')
            self.update_best_model()
            # TB log
            self.write_tb_log()
            # Save checkpoint
            self.save_checkpoint(model)
        self.epoch = 0
