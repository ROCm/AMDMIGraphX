# pip install requests bs4 tqdm
import requests
import logging
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from itertools import chain
from os import makedirs, system, getenv
from os.path import splitext, exists, dirname
from shutil import rmtree
# use TQDM_MININTERVAL=120 for fewer prints
from tqdm import tqdm

# Terminology:
# href: path that start with "/", e.g. /onnx/model.onnx
# url: standard http path

logging.basicConfig(
    format="%(levelname)s|%(funcName)s:%(lineno)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HF_ONNX_URL = "https://huggingface.co/models?library=onnx&sort=downloads&p={p}"
HF_ONNX_URL_IGNORE_LIST = [
    '/', '/datasets', '/docs', '/huggingface', '/join', '/login', '/models',
    '/posts', '/pricing', '/privacy', '/spaces', '/terms-of-service',
    '/fxmarty', '/asifanchor'
]

HF_MODEL_URL = "https://huggingface.co/{url}"
HF_MODEL_URL_SKIP_LIST = [
    '?', 'commit', 'commits', 'discussions', 'LICENSE', '.gitattributes',
    'coreml', '.idea', 'runs'
]
contains_word = lambda s, l: any(map(lambda x: x in s, l))

INT_INPUT_NAMES = [
    "attention_mask",
    "bbox",
    "causal_mask",
    "decoder_input_ids",
    "encoder_attention_mask",
    "encoding_indices",
    "hashed_ids",
    "input_ids",
    "past_sequence_length",
    "position_ids",
    "positions",
    "text",
    "token_type_ids",
]
# Note: It is not official supported to pass names that are not present in the model
FILL1_FLAG = f"--fill1 {' '.join(INT_INPUT_NAMES)}"

DRIVER_PATH = getenv("DRIVER_PATH",
                     "/code/AMDMIGraphX/build/bin/migraphx-driver")
DEFAULT_WORKDIR = "downloaded_models"


def get_soup_from_url(url):
    logger.debug(f"{url = }")
    response = requests.get(url)
    html = response.text
    return BeautifulSoup(html, "html.parser")


def get_hrefs_from_soup(soup, filter_fn=lambda x: True):
    logger.debug(f"{soup.title = }")
    links = soup.find_all("a")
    hrefs = []
    for link in links:
        href = link.get("href")
        if href and filter_fn(href):
            hrefs.append(href)
    return hrefs


def get_model_hrefs_from_onnx_page(page_range):
    logger.debug(f"{page_range = }")
    model_page_hrefs = []
    filter_fn = lambda href: href.startswith("/") and not contains_word(
        href, HF_MODEL_URL_SKIP_LIST) and href not in HF_ONNX_URL_IGNORE_LIST
    for page in page_range:
        soup = get_soup_from_url(HF_ONNX_URL.format(p=page))
        model_page_hrefs.extend(get_hrefs_from_soup(soup, filter_fn))
    return model_page_hrefs


def create_model_url_from_href(href, suffix=""):
    logger.debug(f"{suffix = }")
    return HF_MODEL_URL.format(url=href[1:]) + suffix


def create_model_urls_from_hrefs(hrefs, suffix=""):
    logger.debug(f"{len(hrefs) = } {suffix = }")
    return [create_model_url_from_href(href, suffix) for href in hrefs]


def generate_onnx_hrefs_from_urls(page_urls):
    logger.debug(f"{list(page_urls) = }")
    filter_model_or_data_fn = lambda href: not contains_word(
        href, HF_MODEL_URL_SKIP_LIST) and (href.endswith(
            ".onnx") or href.endswith(".onnx_data") or href.endswith(
                ".onnx.data") or href.endswith("model.data") or href.endswith(
                    ".weight"))
    has_no_extension = lambda s: splitext(s)[1] == ''
    filter_folder_fn = lambda href: href.startswith(
        "/") and "tree/main/" in href and has_no_extension(
            href) and not contains_word(
                href, HF_MODEL_URL_SKIP_LIST
            ) and href not in HF_ONNX_URL_IGNORE_LIST
    filter_non_sub_links_fn = lambda items, prefix: filter(
        lambda x: x.startswith(prefix), items)
    for url in page_urls:
        logger.info(f"Checking {url = }")
        onnx_hrefs = []
        soup = get_soup_from_url(url)
        # get all hrefs for .onnx model and data/weight files
        onnx_hrefs.extend(get_hrefs_from_soup(soup, filter_model_or_data_fn))

        # recursively try to extend the list from folders
        folder_hrefs = generate_onnx_hrefs_from_urls(
            filter_non_sub_links_fn(
                create_model_urls_from_hrefs(
                    get_hrefs_from_soup(soup, filter_folder_fn)), url))
        # flatten result list
        onnx_hrefs.extend(list(chain.from_iterable(folder_hrefs)))
        if not onnx_hrefs:
            logger.warning(f"Not found anything in {url}")
        else:
            yield onnx_hrefs


def download_model_from_href(href, workdir=DEFAULT_WORKDIR, force=False):
    logger.debug(f"{href = } {workdir = } {force = }")
    name = f"{workdir}/{href[1:]}"
    makedirs(dirname(name), exist_ok=True)
    if not force and exists(name):
        logger.info(f"Model already exists at {name}")
        return name

    # blob -> resolve for lfs files
    url = create_model_url_from_href(href).replace('blob', 'resolve')
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(name, mode="wb") as file, tqdm(
            desc=name,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        logger.info(f"Downloading model from {url} to {name}")
        for chunk in response.iter_content(chunk_size=10 * 1024):
            size = file.write(chunk)
            bar.update(size)
    return name


def download_models_from_urls(model_urls,
                              workdir=DEFAULT_WORKDIR,
                              dry_run=False,
                              keep_models=False,
                              force=False):
    logger.debug(
        f"{model_urls = } {workdir = } {dry_run = } {keep_models = } {force = }"
    )
    all_models = []
    for hrefs in generate_onnx_hrefs_from_urls(model_urls):
        logger.info(f"Model(s) found: {','.join(hrefs)}")
        all_models.extend(hrefs)
        if not dry_run:
            try:
                logger.info("Download models")
                # download all related href, because weight can be separate files
                onnx_files = [
                    download_model_from_href(href, workdir, force)
                    for href in hrefs
                ]
                for idx, onnx_file in enumerate(onnx_files):
                    logger.info(f"Start {hrefs[idx] = }")
                    if not onnx_file.endswith(".onnx"):
                        logger.info(f"Skip non-model {onnx_file = }")
                        continue
                    cmds = [
                        f"{DRIVER_PATH} {mode} {dtype} {onnx_file} {FILL1_FLAG}"
                        # for mode in ["compile", "verify"]
                        for mode in ["verify"]  # skip compile
                        for dtype in ["", "--fp16"]
                    ]
                    for cmd in cmds:
                        logger.info(f"Running {cmd = }")
                        system(cmd)
                    logger.info(f"Finish {hrefs[idx] = }")
                # cleanup
            except Exception as e:
                logger.error(f"Something went wrong: {e}")
            finally:
                if not keep_models:
                    logger.info("Cleanup")
                    rmtree(workdir, ignore_errors=True)
                    makedirs(workdir, exist_ok=True)

    return all_models


def main(model_url=None,
         page_range=range(6),
         workdir=DEFAULT_WORKDIR,
         dry_run=False,
         keep_models=False,
         force=False,
         debug=False):
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug(
        f"{model_url = } {page_range = } {workdir = } {dry_run = } {keep_models = } {force = }"
    )
    makedirs(workdir, exist_ok=True)
    if model_url:
        if "/tree/main/" not in model_url:
            model_url += "/tree/main/"
            logger.warn(
                f"Model url should point to the files, extended it {model_url = }"
            )
        model_urls = [model_url]
    else:
        model_urls = create_model_urls_from_hrefs(
            get_model_hrefs_from_onnx_page(page_range=page_range),
            "/tree/main/")
    all_models = download_models_from_urls(model_urls,
                                           workdir=workdir,
                                           dry_run=dry_run,
                                           keep_models=keep_models,
                                           force=force)
    logger.info(f"Found {len(all_models)} model(s).")
    logger.debug(f"{all_models =}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-url",
        type=str,
        default=None,
        help=
        "Hugging Face url for a specific model ('https://huggingface.co/X/Y/tree/main'), if not present Top Downloaded ONNX models will be used."
    )
    parser.add_argument("-w",
                        "--workdir",
                        type=str,
                        default=DEFAULT_WORKDIR,
                        help="Workdir where the models will be downloaded.")
    parser.add_argument(
        "-p",
        "--page-range",
        type=str,
        default="1",
        help=
        "Top models (zero indexed) pages to use 'x' or 'x,y' (python range) e.g. '6' will do the first 6 pages [0, 6), 2,5 will do page 3,4,5 [2,5) (zero indexed)"
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="Only collect urls, but do not download and test them.")
    parser.add_argument("-f",
                        "--force",
                        action="store_true",
                        default=False,
                        help="Re-download any existing models.")
    parser.add_argument(
        "-k",
        "--keep-models",
        action="store_true",
        default=False,
        help=
        "Keep any downloaded model. By default it will be cleaned up to save space"
    )
    parser.add_argument("--debug",
                        action="store_true",
                        default=False,
                        help="Enable debug logging")
    args = parser.parse_args()
    page_ranges = args.page_range.split(
        ',')[:2] if ',' in args.page_range else [args.page_range]
    page_range = range(*map(int, page_ranges))
    args.page_range = page_range
    main(**vars(args))
