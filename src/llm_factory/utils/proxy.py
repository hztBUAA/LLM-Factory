"""
Proxy context manager utilities.
"""

import os
from contextlib import contextmanager
from typing import Dict, Optional


@contextmanager
def ProxyContext(proxy_config: Optional[Dict[str, str]]):
    """
    Context manager for setting proxy environment variables.
    
    Args:
        proxy_config: Dictionary with 'http' and 'https' proxy URLs
    """
    if not proxy_config:
        yield
        return
    
    original_http = os.environ.get('http_proxy')
    original_https = os.environ.get('https_proxy')
    
    try:
        if proxy_config.get('http'):
            os.environ['http_proxy'] = proxy_config['http']
        if proxy_config.get('https'):
            os.environ['https_proxy'] = proxy_config['https']
        
        yield
        
    finally:
        if original_http is not None:
            os.environ['http_proxy'] = original_http
        else:
            os.environ.pop('http_proxy', None)
            
        if original_https is not None:
            os.environ['https_proxy'] = original_https
        else:
            os.environ.pop('https_proxy', None)
