#!/usr/bin/env python3
"""
Simple script to inspect the SQLite database values.
This script provides a command-line interface to explore the database contents.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from sqlitedict import SqliteDict


class DatabaseInspector:
    """Simple database inspector for the training responses database."""
    
    def __init__(self, db_path: str):
        """Initialize the database inspector."""
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Open database connections
        self.responses_db = SqliteDict(db_path, tablename='responses', flag='r')
        self.queries_db = SqliteDict(db_path, tablename='queries', flag='r')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic database statistics."""
        # Get successful responses statistics
        successful_stats = self.get_successful_responses_stats()

        return {
            'db_path': self.db_path,
            'db_size_mb': round(os.path.getsize(self.db_path) / (1024 * 1024), 2),
            'num_responses': len(self.responses_db),
            'num_queries': len(self.queries_db),
            'queries_with_success': successful_stats.get('total_queries_with_success', 0),
            'total_successful_responses': successful_stats.get('total_successful_responses', 0),
            'success_rate_percent': successful_stats.get('success_rate_percent', 0),
        }
    
    def list_tables(self):
        """List all tables and their sizes."""
        print("Database Tables:")
        print(f"  responses: {len(self.responses_db)} entries")
        print(f"  queries: {len(self.queries_db)} entries")
    
    def show_sample_response(self, limit: int = 1):
        """Show a sample response entry."""
        print(f"\nSample Response Entry (showing {limit}):")
        count = 0
        for key, value in self.responses_db.items():
            print(f"\nKey: {key}")
            print("Data:")
            for k, v in value.items():
                if isinstance(v, (list, dict)) and len(str(v)) > 100:
                    print(f"  {k}: {type(v).__name__} (length: {len(v)})")
                else:
                    print(f"  {k}: {v}")
            count += 1
            if count >= limit:
                break
    
    def show_sample_query(self, limit: int = 1):
        """Show a sample query entry."""
        print(f"\nSample Query Entry (showing {limit}):")
        count = 0
        for key, value in self.queries_db.items():
            print(f"\nQuery ID: {key}")
            print("Data:")
            for k, v in value.items():
                if k == 'successful_responses' and v:
                    print(f"  {k}: {len(v)} successful responses")
                    for i, resp in enumerate(v[:3]):  # Show first 3 successful responses
                        print(f"    [{i}] Response ID: {resp['response_id']}, Length: {resp['response_length']}")
                    if len(v) > 3:
                        print(f"    ... and {len(v) - 3} more")
                else:
                    print(f"  {k}: {v}")
            count += 1
            if count >= limit:
                break
    
    def list_response_keys(self, limit: int = 10):
        """List response keys."""
        print(f"\nResponse Keys (showing first {limit}):")
        for i, key in enumerate(self.responses_db.keys()):
            if i >= limit:
                print(f"... and {len(self.responses_db) - limit} more")
                break
            print(f"  {key}")
    
    def list_query_ids(self, limit: int = 10):
        """List query IDs."""
        print(f"\nQuery IDs (showing first {limit}):")
        for i, key in enumerate(self.queries_db.keys()):
            if i >= limit:
                print(f"... and {len(self.queries_db) - limit} more")
                break
            print(f"  {key}")
    
    def get_response_by_key(self, key: str) -> Optional[Dict]:
        """Get a specific response by key."""
        return self.responses_db.get(key)
    
    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """Get a specific query by ID."""
        return self.queries_db.get(query_id)
    
    def search_responses(self, **filters) -> List[Dict]:
        """Search responses with filters."""
        results = []
        for key, response in self.responses_db.items():
            match = True
            for filter_key, filter_value in filters.items():
                if filter_key not in response or response[filter_key] != filter_value:
                    match = False
                    break
            if match:
                results.append({'key': key, 'data': response})
        return results
    
    def get_unique_values(self, field: str, table: str = 'responses') -> List:
        """Get unique values for a field."""
        db = self.responses_db if table == 'responses' else self.queries_db
        values = set()
        for entry in db.values():
            if field in entry:
                values.add(entry[field])
        return sorted(list(values))

    def show_queries_with_successful_responses(self, limit: int = 10):
        """Show queries that have successful responses."""
        print(f"\nQueries with Successful Responses (showing first {limit}):")
        count = 0
        for query_id, query_data in self.queries_db.items():
            successful_responses = query_data.get('successful_responses', [])
            if successful_responses:
                print(f"\nQuery ID: {query_id}")
                print(f"  Query: {query_data.get('query', 'N/A')[:100]}{'...' if len(query_data.get('query', '')) > 100 else ''}")
                print(f"  Successful Responses: {len(successful_responses)}")
                print(f"  Average Response Length: {sum(r['response_length'] for r in successful_responses) / len(successful_responses):.1f} tokens")
                print(f"  Response IDs: {[r['response_id'] for r in successful_responses[:3]]}")
                if len(successful_responses) > 3:
                    print(f"    ... and {len(successful_responses) - 3} more")
                count += 1
                if count >= limit:
                    break

    def get_successful_responses_stats(self) -> Dict[str, Any]:
        """Get statistics about successful responses."""
        total_queries_with_success = 0
        total_successful_responses = 0
        all_response_lengths = []

        for query_data in self.queries_db.values():
            successful_responses = query_data.get('successful_responses', [])
            if successful_responses:
                total_queries_with_success += 1
                total_successful_responses += len(successful_responses)
                all_response_lengths.extend([r['response_length'] for r in successful_responses])

        stats = {
            'total_queries_with_success': total_queries_with_success,
            'total_successful_responses': total_successful_responses,
            'success_rate_percent': (total_queries_with_success / len(self.queries_db)) * 100 if self.queries_db else 0,
        }

        if all_response_lengths:
            stats.update({
                'avg_response_length': sum(all_response_lengths) / len(all_response_lengths),
                'min_response_length': min(all_response_lengths),
                'max_response_length': max(all_response_lengths),
                'median_response_length': sorted(all_response_lengths)[len(all_response_lengths) // 2]
            })

        return stats

    def show_detailed_successful_responses(self, query_id: str):
        """Show detailed information about successful responses for a specific query."""
        query_data = self.queries_db.get(query_id)
        if not query_data:
            print(f"No query found for ID: {query_id}")
            return

        successful_responses = query_data.get('successful_responses', [])
        if not successful_responses:
            print(f"No successful responses found for query ID: {query_id}")
            return

        print(f"\nDetailed Successful Responses for Query ID: {query_id}")
        print(f"Query: {query_data.get('query', 'N/A')}")
        print(f"Total Successful Responses: {len(successful_responses)}")
        print("\nSuccessful Response Details:")

        for i, resp in enumerate(successful_responses):
            response_id = resp['response_id']
            response_length = resp['response_length']

            # Try to get the actual response data
            response_data = self.responses_db.get(response_id)
            if response_data:
                print(f"\n[{i+1}] Response ID: {response_id}")
                print(f"    Length: {response_length} tokens")
                print(f"    Score: {response_data.get('score', 'N/A')}")
                print(f"    Dataset: {response_data.get('dataset', 'N/A')}")
                print(f"    Training Step: {response_data.get('training_step', 'N/A')}")
                print(f"    Response Text Preview: {response_data.get('response_text', 'N/A')[:100]}{'...' if len(response_data.get('response_text', '')) > 100 else ''}")
            else:
                print(f"\n[{i+1}] Response ID: {response_id} (data not found)")
                print(f"    Length: {response_length} tokens")

    def close(self):
        """Close database connections."""
        self.responses_db.close()
        self.queries_db.close()


def main():
    parser = argparse.ArgumentParser(description='Inspect SQLite database contents')
    parser.add_argument('--db', '-d', default='../output/dbs/Qwen3_4B_Instruct_ballsim_G.db', 
                       help='Path to database file (default: training_responses.db)')
    parser.add_argument('--stats', action='store_true', 
                       help='Show database statistics')
    parser.add_argument('--tables', action='store_true', 
                       help='List tables and their sizes')
    parser.add_argument('--sample-response', type=int, metavar='N', 
                       help='Show N sample response entries')
    parser.add_argument('--sample-query', type=int, metavar='N', 
                       help='Show N sample query entries')
    parser.add_argument('--list-keys', type=int, metavar='N', 
                       help='List N response keys')
    parser.add_argument('--list-queries', type=int, metavar='N', 
                       help='List N query IDs')
    parser.add_argument('--get-response', metavar='KEY', 
                       help='Get specific response by key')
    parser.add_argument('--get-query', metavar='ID', 
                       help='Get specific query by ID')
    parser.add_argument('--search', nargs=2, metavar=('FIELD', 'VALUE'), action='append',
                       help='Search responses by field=value (can be used multiple times)')
    parser.add_argument('--unique', metavar='FIELD',
                       help='Get unique values for a field in responses table')
    parser.add_argument('--successful-queries', type=int, metavar='N',
                       help='Show N queries that have successful responses')
    parser.add_argument('--successful-stats', action='store_true',
                       help='Show statistics about successful responses')
    parser.add_argument('--detailed-success', metavar='QUERY_ID',
                       help='Show detailed successful responses for a specific query ID')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Default behavior if no arguments
    if len(sys.argv) == 1:
        args.stats = True
        args.tables = True
        args.sample_response = 1
        args.sample_query = 1
    
    try:
        inspector = DatabaseInspector(args.db)
        
        if args.stats:
            stats = inspector.get_stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print("Database Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
        
        if args.tables:
            if not args.json:
                inspector.list_tables()
        
        if args.sample_response:
            if not args.json:
                inspector.show_sample_response(args.sample_response)
        
        if args.sample_query:
            if not args.json:
                inspector.show_sample_query(args.sample_query)
        
        if args.list_keys:
            if not args.json:
                inspector.list_response_keys(args.list_keys)
        
        if args.list_queries:
            if not args.json:
                inspector.list_query_ids(args.list_queries)
        
        if args.get_response:
            response = inspector.get_response_by_key(args.get_response)
            if args.json:
                print(json.dumps(response, indent=2, default=str))
            else:
                if response:
                    print(f"\nResponse for key '{args.get_response}':")
                    for k, v in response.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"No response found for key: {args.get_response}")
        
        if args.get_query:
            query = inspector.get_query_by_id(args.get_query)
            if args.json:
                print(json.dumps(query, indent=2, default=str))
            else:
                if query:
                    print(f"\nQuery for ID '{args.get_query}':")
                    for k, v in query.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"No query found for ID: {args.get_query}")
        
        if args.search:
            filters = {field: value for field, value in args.search}
            results = inspector.search_responses(**filters)
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            else:
                print(f"\nSearch results for {filters} ({len(results)} found):")
                for result in results[:10]:  # Limit to first 10
                    print(f"  Key: {result['key']}")
                    for k, v in result['data'].items():
                        if isinstance(v, (list, dict)) and len(str(v)) > 100:
                            print(f"    {k}: {type(v).__name__} (length: {len(v)})")
                        else:
                            print(f"    {k}: {v}")
                    print()
        
        if args.unique:
            values = inspector.get_unique_values(args.unique)
            if args.json:
                print(json.dumps(values, indent=2))
            else:
                print(f"\nUnique values for field '{args.unique}' ({len(values)} total):")
                for value in values:
                    print(f"  {value}")

        if args.successful_queries:
            if not args.json:
                inspector.show_queries_with_successful_responses(args.successful_queries)

        if args.successful_stats:
            stats = inspector.get_successful_responses_stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print("\nSuccessful Responses Statistics:")
                for k, v in stats.items():
                    if 'percent' in k:
                        print(f"  {k}: {v:.2f}%")
                    elif isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")

        if args.detailed_success:
            if not args.json:
                inspector.show_detailed_successful_responses(args.detailed_success)

        inspector.close()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
