labels = {
    "1": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic",
        "bmp", "gif", "png", "ai", "eps", "psd", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar",
        "msi", "dmg", "7z", "bz2", "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll",
        "doc", "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi", "pdf", "md", "rtf", "txt",
        "tex", "json", "html", "xml", "log", "csv", "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap",
        "ttf", "dwg", "sqlite"],
    "2": ["bmp", "raw", "vec", "vid", "arc", "exe", "off", "pub", "hr", "aud", "oth"],
    "3": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic",
        "bmp", "gif", "png", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "oth"],
    "4": ["jpg", "raw", "vid", "5_bmps", "oth"],
    "5": ["jpg", "oth"],
    "6": ["jpg", "oth"],
    "tags": ["bitmap", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "bitmap",
            "bitmap", "bitmap", "bitmap", "bitmap", "vector", "vector", "vector", "video", "video", "video",
            "video", "video", "video", "video", "archive", "archive", "archive", "archive", "archive", "archive",
            "archive", "archive", "archive", "archive", "archive", "archive", "archive", "executable",
            "executable", "executable", "executable", "office", "office", "office", "office", "office", "office",
            "office", "published", "published", "published", "published", "human-readable", "human-readable",
            "human-readable", "human-readable", "human-readable", "human-readable", "human-readable",
            "human-readable", "human-readable", "audio", "audio", "audio", "audio", "audio", "audio", "audio",
            "misc", "misc", "misc", "misc"]}

BLOCK_SIZE = 512
SCENARIO = '1'
DIM = [4, 8, 16, 32]
MAX_CLASSES = len(labels[SCENARIO])
DATASET_PATH = '/home/wyb/zbl'
CHECKPOINT_PATH = '/home/wyb/zbl/checkpoints/'
EPOCHS = 50
BATCH_SIZE = 128
TOP_LR = 1e-4
HEATING_EPOCHS = 5
COOLING_EPOCHS = 20
BOTTOM_LR = 1e-6
WEIGHT_DECAY = 0.02
BETAS = (0.9, 0.999)
SEED = 42
DEVICE = 'cuda'
CONTINUE_FROM_CHECKPOINT = False
CHECKPOINT_NAME = 'best_model.pth'
PATIENCE = 5

