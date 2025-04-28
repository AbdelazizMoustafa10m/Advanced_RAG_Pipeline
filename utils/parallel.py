# --- utils/parallel.py ---

import os
import concurrent.futures
from typing import List, Callable, Any, Optional, TypeVar, Generic
import logging
from functools import partial

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def parallel_process(
    items: List[T],
    process_func: Callable[[T], R],
    max_workers: Optional[int] = None,
    func_kwargs: Optional[dict] = None
) -> List[R]:
    """Process items in parallel.
    
    Args:
        items: List of items to process
        process_func: Function to process each item
        max_workers: Maximum number of worker processes
        func_kwargs: Additional keyword arguments for process_func
        
    Returns:
        List of processed results
    """
    if not items:
        return []
    
    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4)
    
    if func_kwargs is None:
        func_kwargs = {}
    
    # Create partial function with additional kwargs
    if func_kwargs:
        process_func = partial(process_func, **func_kwargs)
    
    logger.info(f"Processing {len(items)} items with {max_workers} workers")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all items for processing
        future_to_item = {executor.submit(process_func, item): item for item in items}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {str(e)}")
    
    return results


def batch_process(
    items: List[T],
    process_func: Callable[[List[T]], List[R]],
    batch_size: int = 10,
    max_workers: Optional[int] = None
) -> List[R]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to process a batch of items
        batch_size: Number of items per batch
        max_workers: Maximum number of worker processes
        
    Returns:
        List of processed results
    """
    if not items:
        return []
    
    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4)
    
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    logger.info(f"Processing {len(items)} items in {len(batches)} batches with {max_workers} workers")
    
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches for processing
        future_to_batch = {executor.submit(process_func, batch): i for i, batch in enumerate(batches)}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                logger.info(f"Completed batch {batch_idx + 1}/{len(batches)}")
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
    
    return all_results
