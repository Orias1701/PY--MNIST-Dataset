import os
import re
import json
import pandas as pd
from . import Common_MyUtils as MyUtils

## ========================================
## Helpers
## ========================================

def get_urls_from_url_file(file_path):
    """Lấy set các URL đã có từ file URLS.json."""
    urls = set()
    url_info_list = MyUtils.read_json(file_path) 
    for item in url_info_list:
        if 'url' in item:
            urls.add(item['url'])
    return urls

def get_existing_article_urls(file_path):
    """Lấy set các URL bài viết đã có từ file JSONL."""
    urls = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    urls.add(json.loads(line)['url'])
                except (json.JSONDecodeError, KeyError):
                    continue
    return urls

def get_url_key(item):
    match = re.search(r'-(\d+)\.html', item['url'])
    return int(match.group(1)) if match else 0

def heapify(arr, n, i, key_func):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and key_func(arr[l]) > key_func(arr[largest]): largest = l
    if r < n and key_func(arr[r]) > key_func(arr[largest]): largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, key_func)

def heapSort(arr, key_func):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, key_func)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, key_func)
    return arr