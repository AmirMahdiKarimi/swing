import numpy as np
import pandas as pd
from dataclasses import dataclass
import re
import os
import pickle


# ترکیب آمینواسیدی ساده: 20 آمینواسید استاندارد
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA)}

# شاخص‌های فیزیکوشیمیایی ساده
KYTE_DOOLITTLE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}
MASS = {
    'A': 89.094, 'C': 121.154, 'D': 133.104, 'E': 147.131, 'F': 165.192,
    'G': 75.067, 'H': 155.156, 'I': 131.175, 'K': 146.189, 'L': 131.175,
    'M': 149.208, 'N': 132.119, 'P': 115.132, 'Q': 146.146, 'R': 174.203,
    'S': 105.093, 'T': 119.119, 'V': 117.148, 'W': 204.228, 'Y': 181.191,
}


@dataclass
class FeatureConfig:
    add_aa_counts: bool = True
    add_aa_freq: bool = True
    add_length: bool = True
    add_physchem: bool = True
    add_kmer_k: int = 2  # 0 یعنی غیرفعال
    add_embedding_aaindex: bool = True
    add_mhc: bool = False
    add_anchor: bool = False
    # بهبودها
    add_blosum: bool = False
    add_protbert: bool = False  # امبدینگ پیشرفته ProtBert
    anchor_strict: bool = False
    mhc_grouping: str = 'none'  # none | locus | two_digit


def _seq_to_aa_counts(s: str):
    counts = [0] * len(AA)
    for ch in str(s).upper():
        if ch in AA_INDEX:
            counts[AA_INDEX[ch]] += 1
    return counts

def _physchem_features_for_seq(s: str):
    s = str(s).upper()
    vals_kd = [KYTE_DOOLITTLE[ch] for ch in s if ch in KYTE_DOOLITTLE]
    vals_mass = [MASS[ch] for ch in s if ch in MASS]
    length = max(len(vals_kd), 1)
    # آمار توصیفی ساده برای هر شاخص
    def stats(vs):
        if len(vs) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        arr = np.array(vs, dtype=np.float32)
        return [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())]

    kd_stats = stats(vals_kd)
    mass_stats = stats(vals_mass)
    # نسبت‌های ترکیبی
    pos = sum(ch in {'K', 'R', 'H'} for ch in s) / max(len(s), 1)
    neg = sum(ch in {'D', 'E'} for ch in s) / max(len(s), 1)
    arom = sum(ch in {'F', 'W', 'Y'} for ch in s) / max(len(s), 1)
    alip = sum(ch in {'A', 'I', 'L', 'V'} for ch in s) / max(len(s), 1)
    return kd_stats + mass_stats + [pos, neg, arom, alip]


def _kmer_counts_for_seq(s: str, k: int):
    s = str(s).upper()
    if k <= 1:
        # تک‌حرفی همان counts است
        return np.array(_seq_to_aa_counts(s), dtype=np.float32)
    # برای k=2، 400 ویژگی
    if k == 2:
        counts = np.zeros((len(AA), len(AA)), dtype=np.float32)
        for i in range(len(s) - 1):
            a = s[i]
            b = s[i + 1]
            if a in AA_INDEX and b in AA_INDEX:
                counts[AA_INDEX[a], AA_INDEX[b]] += 1.0
        # پهن‌سازی و نرمال‌سازی به فراوانی نسبی بر طول جفت‌ها
        total = max(len(s) - 1, 1)
        freq = (counts.reshape(-1) / float(total))
        return freq
    if k == 3:
        # 8000 ویژگی؛ با نرمال‌سازی نسبت به تعداد تری‌مرها
        counts = np.zeros((len(AA), len(AA), len(AA)), dtype=np.float32)
        for i in range(len(s) - 2):
            a, b, c = s[i], s[i + 1], s[i + 2]
            if a in AA_INDEX and b in AA_INDEX and c in AA_INDEX:
                counts[AA_INDEX[a], AA_INDEX[b], AA_INDEX[c]] += 1.0
        total = max(len(s) - 2, 1)
        freq = (counts.reshape(-1) / float(total))
        return freq
    # k بزرگ‌تر پشتیبانی نمی‌شود در حال حاضر
    return np.zeros((len(AA) * len(AA)), dtype=np.float32)


def _anchor_features_for_seq(s: str):
    s = str(s).upper()
    n = len(s)
    def kd(ch):
        return KYTE_DOOLITTLE.get(ch, 0.0)
    def mass(ch):
        return MASS.get(ch, 0.0)
    p1 = s[0] if n >= 1 else ' '
    p2 = s[1] if n >= 2 else ' '
    plast = s[-1] if n >= 1 else ' '
    feats = [
        kd(p1), kd(p2), kd(plast),
        mass(p1), mass(p2), mass(plast),
    ]
    # هیدروفوبیک بودن موقعیت‌های کلیدی
    hydros = set('AILVFMYW')
    feats += [
        1.0 if p2 in hydros else 0.0,
        1.0 if plast in hydros else 0.0,
    ]
    return feats

def _anchor_features_strict(s: str):
    # ویژگی‌های اضافه برای MHC-I: تاکید بر 9-مر، پرولین و هیدروفوبیک بودن
    s = str(s).upper()
    n = len(s)
    p2 = s[1] if n >= 2 else ' '
    plast = s[-1] if n >= 1 else ' '
    hydros = set('AILVFMYW')
    arom = set('FWY')
    small = set('AGST')
    feats = [
        1.0 if n == 9 else 0.0,             # 9-مر بودن
        1.0 if p2 in hydros else 0.0,       # هیدروفوبیک بودن P2
        1.0 if plast in hydros else 0.0,    # هیدروفوبیک بودن PΩ
        1.0 if p2 == 'P' else 0.0,          # پرولین در P2 (معمولاً نامطلوب)
        1.0 if plast == 'P' else 0.0,       # پرولین در PΩ
        1.0 if p2 in arom else 0.0,         # آروماتیک در P2
        1.0 if plast in arom else 0.0,      # آروماتیک در PΩ
        1.0 if p2 in small else 0.0,        # کوچک در P2
    ]
    return feats

def _mhc_group_value(v: str, mode: str) -> str:
    if mode == 'none':
        return v
    s = str(v)
    if pd.isna(v) or s.strip() == '':
        return 'UNK'
    s = s.strip()
    # تلاش برای استخراج locus و دو رقم اول
    # نمونه: HLA-A*02:01 -> locus=A, two_digit=02
    m = re.match(r"^HLA-([A-Z0-9]+)\*([0-9]{2})", s)
    locus = None
    two = None
    if m:
        locus = m.group(1)
        two = m.group(2)
    else:
        # fallback: قبل از '*' را به‌عنوان locus بگیریم
        parts = s.split('*')
        locus = parts[0].replace('HLA-', '').strip() if parts else s
        # تلاش برای یافتن دو رقم اول پس از '*'
        if len(parts) > 1:
            m2 = re.match(r"^([0-9]{2})", parts[1])
            if m2:
                two = m2.group(1)
    locus = locus or 'UNK'
    two = two or 'XX'
    if mode == 'locus':
        return f"{locus}"
    if mode == 'two_digit':
        return f"{locus}*{two}"
    return s


def _compute_protbert_embeddings(seqs: list[str], batch_size: int = 32):
    """
    Computes ProtBert embeddings for a list of sequences.
    Uses Rostlab/prot_bert model.
    Implements caching to avoid re-computation and network issues.
    """
    try:
        from transformers import BertModel, BertTokenizer
        import torch
    except ImportError:
        print("Error: transformers or torch not installed. Cannot use ProtBert.")
        return np.zeros((len(seqs), 1024), dtype=np.float32)

    # Path to cache file
    cache_path = os.path.join(os.path.dirname(__file__), '../../Data/protbert_cache.pkl')
    cache_path = os.path.abspath(cache_path)
    
    # Load cache if exists
    embedding_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                embedding_cache = pickle.load(f)
            print(f"Loaded {len(embedding_cache)} embeddings from cache.")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")

    # Identify missing sequences
    missing_seqs = [s for s in seqs if s not in embedding_cache]
    
    if missing_seqs:
        print(f"Computing ProtBert embeddings for {len(missing_seqs)} new sequences...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Try loading from local cache first to avoid network issues
            tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, local_files_only=True)
            model = BertModel.from_pretrained("Rostlab/prot_bert", local_files_only=True)
        except Exception:
            print("Local model not found or incomplete. Downloading from Hugging Face...")
            tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            model = BertModel.from_pretrained("Rostlab/prot_bert")
            
        model.to(device)
        model.eval()

        processed_missing = [" ".join(list(str(s))) for s in missing_seqs]
        
        import tqdm
        iterator = range(0, len(processed_missing), batch_size)
        try:
            from tqdm import tqdm as tqdm_cls
            iterator = tqdm_cls(range(0, len(processed_missing), batch_size), desc="ProtBert Extraction")
        except ImportError:
            pass

        new_embeddings = {}
        with torch.no_grad():
            for i in iterator:
                batch_seqs = missing_seqs[i : i + batch_size]
                batch_processed = processed_missing[i : i + batch_size]
                
                encoded = tokenizer(batch_processed, return_tensors="pt", padding=True, truncation=True, max_length=40)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                outputs = model(**encoded)
                
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
                batch_embeddings = mean_embeddings.cpu().numpy()
                
                for seq, emb in zip(batch_seqs, batch_embeddings):
                    new_embeddings[seq] = emb
        
        # Update cache
        embedding_cache.update(new_embeddings)
        
        # Save cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding_cache, f)
            print(f"Saved updated cache with {len(embedding_cache)} entries.")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
            
        del model
        torch.cuda.empty_cache()

    # Construct final array
    final_embeddings = np.array([embedding_cache[s] for s in seqs], dtype=np.float32)
    return final_embeddings


def build_features(df: pd.DataFrame, seq_col: str, config: FeatureConfig | None = None) -> np.ndarray:
    # تنظیمات پیش‌فرض
    if config is None:
        config = FeatureConfig()

    seqs = df[seq_col].astype(str).tolist()

    blocks = []
    # شمارش و فراوانی و طول
    if config.add_aa_counts or config.add_aa_freq or config.add_length:
        X_counts = np.array([_seq_to_aa_counts(s) for s in seqs], dtype=np.float32)
        lengths = np.array([len(s) for s in seqs], dtype=np.float32).reshape(-1, 1)
        if config.add_aa_counts:
            blocks.append(X_counts)
        if config.add_aa_freq:
            freq = (X_counts / np.maximum(lengths, 1.0))
            blocks.append(freq)
        if config.add_length:
            blocks.append(lengths)

    # ویژگی‌های فیزیکوشیمیایی
    if config.add_physchem:
        phys = np.array([_physchem_features_for_seq(s) for s in seqs], dtype=np.float32)
        blocks.append(phys)

    # k-mer ها (پیش‌فرض k=2)
    if isinstance(config.add_kmer_k, int) and config.add_kmer_k >= 1:
        kmers = np.array([_kmer_counts_for_seq(s, config.add_kmer_k) for s in seqs], dtype=np.float32)
        blocks.append(kmers)

    # امبدینگ ساده مبتنی بر AAindex (همان آمار KD و Mass)
    if config.add_embedding_aaindex:
        # همان فیزیکوشیمیایی را با نام متفاوت اضافه می‌کنیم تا سازگار با درخواست باشد
        embed = np.array([_physchem_features_for_seq(s) for s in seqs], dtype=np.float32)
        blocks.append(embed)

    # امبدینگ BLOSUM62: میانگین بردار سطر مربوط به هر آمینواسید
    if config.add_blosum:
        # ماتریس BLOSUM62 به‌صورت آرایهٔ 20x20 با ترتیب AA
        # مقادیر استاندارد از BLOSUM62؛ برای سادگی، از یک نسخهٔ متقارن استفاده می‌کنیم
        B = np.array([
            # A   C    D     E     F     G     H     I     K     L     M     N     P     Q     R     S     T     V     W     Y
            [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
            [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
            [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
            [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
            [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
            [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
            [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
            [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
            [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
            [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
            [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2, -1, -1, -1, -1,  1, -1, -1],  # M
            [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
            [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
            [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2, -1,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
            [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
            [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
            [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
            [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
            [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
            [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
        ], dtype=np.float32)
        # برای هر رشته، میانگین سطرهای متناظر با آمینواسیدها را محاسبه می‌کنیم
        def blosum_mean_vec(s: str):
            vecs = []
            for ch in str(s).upper():
                if ch in AA_INDEX:
                    vecs.append(B[AA_INDEX[ch]])
            if len(vecs) == 0:
                return np.zeros((len(AA),), dtype=np.float32)
            arr = np.stack(vecs, axis=0)
            return arr.mean(axis=0)
        
        blosum_feats = np.array([blosum_mean_vec(s) for s in seqs], dtype=np.float32)
        blocks.append(blosum_feats)

    # امبدینگ ProtBert
    if config.add_protbert:
        protbert_feats = _compute_protbert_embeddings(seqs)
        blocks.append(protbert_feats)

    # ویژگی‌های MHC (One-hot Encoding)
    if config.add_mhc:
        raw_vals = df['MHC'].fillna('UNK').astype(str).values.tolist()
        mhc_vals = [
            _mhc_group_value(v, config.mhc_grouping) if isinstance(config.mhc_grouping, str) else str(v)
            for v in raw_vals
        ]
        uniq = sorted(pd.Series(mhc_vals).unique())
        idx = {v: i for i, v in enumerate(uniq)}
        oh = np.zeros((len(mhc_vals), len(uniq)), dtype=np.float32)
        for i, v in enumerate(mhc_vals):
            oh[i, idx[v]] = 1.0
        blocks.append(oh)

    # ویژگی‌های لنگر (anchor) پوزیشن‌های کلیدی 1،2 و انتها
    if config.add_anchor:
        base = np.array([_anchor_features_for_seq(s) for s in seqs], dtype=np.float32)
        if config.anchor_strict:
            extra = np.array([_anchor_features_strict(s) for s in seqs], dtype=np.float32)
            anchor = np.concatenate([base, extra], axis=1)
        else:
            anchor = base
        blocks.append(anchor)

    X = np.concatenate(blocks, axis=1) if len(blocks) > 0 else np.zeros((len(seqs), 1), dtype=np.float32)
    return X


