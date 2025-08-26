# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

from sqlitedict import SqliteDict


class SQLiteLogger:
    """
    A logger that stores training responses and metrics in a SQLite database using sqlitedict.
    
    This class maintains separate tables for:
    - responses: Individual response data with query_id, rollout_id, scores, metrics, etc.
    - queries: Mapping between query_id and the actual query text
    """
    
    def __init__(
        self,
        db_path: str,
        enabled: bool = True,
        autocommit: bool = True,
        journal_mode: str = "WAL",
    ):
        """
        Initialize the SQLite logger.
        
        Args:
            db_path: Path to the SQLite database file
            enabled: Whether logging is enabled
            autocommit: Whether to autocommit transactions
            journal_mode: SQLite journal mode (WAL recommended for concurrent access)
        """
        self.db_path = db_path
        self.enabled = enabled
        self.autocommit = autocommit
        self.journal_mode = journal_mode
        
        # Cache for query_id to query mapping to avoid duplicate storage
        self._query_cache = set()
        
        if self.enabled:
            self._init_db()
    
    def _init_db(self):
        """Initialize the database tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create/open the responses table
        self.responses_db = SqliteDict(
            self.db_path,
            tablename='responses',
            autocommit=self.autocommit,
            journal_mode=self.journal_mode
        )
        
        # Create/open the queries table  
        self.queries_db = SqliteDict(
            self.db_path,
            tablename='queries',
            autocommit=self.autocommit,
            journal_mode=self.journal_mode
        )
        
        # Load existing query IDs into cache to avoid duplicates
        try:
            self._query_cache = set(self.queries_db.keys())
        except Exception:
            # If table doesn't exist yet, start with empty cache
            self._query_cache = set()
    
    def _get_query_id(self, query: str) -> str:
        """
        Generate a hash-based query_id for a given query.
        
        Args:
            query: The query string
            
        Returns:
            A hash-based string identifier for the query
        """
        # Use SHA-256 hash of the query text
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
    
    def _ensure_query_stored(self, query_id: str, query: str):
        """
        Ensure the query is stored in the queries table.
        
        Args:
            query_id: The query identifier
            query: The actual query text
        """
        if query_id not in self._query_cache:
            self.queries_db[query_id] = {
                'query': query,
                'created_at': time.time(),
                'created_at_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }
            self._query_cache.add(query_id)
    
    def log_responses(
        self,
        training_step: int,
        epoch_id: float,
        queries: List[str],
        responses: List[List[int]],
        decoded_responses: List[str],
        scores: List[float],
        ground_truths: List[Union[str, List[str]]],
        datasets: List[str],
        finish_reasons: List[str],
        advantages: Optional[List[float]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        infos: Optional[List[Any]] = None,
        num_samples_per_prompt_rollout: int = 1,
        **additional_data: Any
    ):
        """
        Log response data to the SQLite database.
        
        Args:
            training_step: Current training step
            epoch_id: Current epoch (can be fractional)
            queries: List of query strings
            responses: List of response token IDs
            decoded_responses: List of decoded response strings
            scores: List of scores for each response
            ground_truths: List of ground truth answers
            datasets: List of dataset sources
            finish_reasons: List of finish reasons
            advantages: Optional list of advantage values
            metrics: Optional dictionary of metrics
            infos: Optional additional info (tool usage, etc.)
            num_samples_per_prompt_rollout: Number of samples per prompt
            **additional_data: Any additional data to store
        """
        if not self.enabled:
            return
        
        timestamp = time.time()
        timestamp_iso = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        # Calculate rollout_id based on position within each prompt group
        for i, (query, response, decoded_response, score, ground_truth, dataset, finish_reason) in enumerate(
            zip(queries, responses, decoded_responses, scores, ground_truths, datasets, finish_reasons)
        ):
            # Calculate which prompt this response belongs to and its rollout index
            prompt_idx = i // num_samples_per_prompt_rollout
            rollout_id = i % num_samples_per_prompt_rollout
            
            # Get query_id and ensure query is stored
            query_id = self._get_query_id(query)
            self._ensure_query_stored(query_id, query)
            
            # Prepare response data
            response_data = {
                'query_id': query_id,
                'training_step': training_step,
                'epoch_id': epoch_id,
                'rollout_id': rollout_id,
                'prompt_idx': prompt_idx,
                'response_tokens': response,
                'response_text': decoded_response,
                'score': score,
                'ground_truth': ground_truth,
                'dataset': dataset,
                'finish_reason': finish_reason,
                'timestamp': timestamp,
                'timestamp_iso': timestamp_iso,
            }
            
            # Add optional data
            if advantages is not None and i < len(advantages):
                response_data['advantage'] = advantages[i]
            
            if infos is not None and i < len(infos):
                response_data['info'] = infos[i]
            
            # Add metrics (same for all responses in this batch)
            if metrics:
                response_data['metrics'] = metrics
            
            # Add any additional data
            response_data.update(additional_data)
            
            # Generate unique key for this response
            response_key = f"{training_step}_{prompt_idx}_{rollout_id}_{timestamp}"
            
            # Store the response
            self.responses_db[response_key] = response_data
    
    def get_responses(
        self,
        training_step: Optional[int] = None,
        query_id: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve responses from the database with optional filtering.
        
        Args:
            training_step: Filter by training step
            query_id: Filter by query ID
            dataset: Filter by dataset
            limit: Maximum number of responses to return
            
        Returns:
            List of response dictionaries
        """
        if not self.enabled:
            return []
        
        responses = []
        count = 0
        
        for key, response_data in self.responses_db.items():
            # Apply filters
            if training_step is not None and response_data.get('training_step') != training_step:
                continue
            if query_id is not None and response_data.get('query_id') != query_id:
                continue
            if dataset is not None and response_data.get('dataset') != dataset:
                continue
            
            responses.append(response_data)
            count += 1
            
            if limit is not None and count >= limit:
                break
        
        return responses
    
    def get_query(self, query_id: str) -> Optional[str]:
        """
        Retrieve the original query text for a given query_id.
        
        Args:
            query_id: The query identifier
            
        Returns:
            The original query text, or None if not found
        """
        if not self.enabled:
            return None
        
        query_data = self.queries_db.get(query_id)
        return query_data['query'] if query_data else None
    
    def get_all_queries(self) -> Dict[str, str]:
        """
        Get all query_id to query mappings.
        
        Returns:
            Dictionary mapping query_id to query text
        """
        if not self.enabled:
            return {}
        
        return {query_id: data['query'] for query_id, data in self.queries_db.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'db_path': self.db_path,
            'num_responses': len(self.responses_db),
            'num_queries': len(self.queries_db),
            'db_size_bytes': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
        }
    
    def close(self):
        """Close the database connections."""
        if self.enabled:
            if hasattr(self, 'responses_db'):
                self.responses_db.close()
            if hasattr(self, 'queries_db'):
                self.queries_db.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
