#!/usr/bin/env python3
"""
Flask web application for viewing SQLite training database.
Provides a web interface to explore training responses and queries with Bootstrap styling.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, abort
from typing import Dict, List, Any, Optional
import json
from urllib.parse import urlencode

# Add parent directory to path to import inspect_db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_db import DatabaseInspector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
os.environ["DB_PATH"] = "../output/dbs/Qwen3_4B_Instruct_ballsim_G.db"  # Default DB path

# Add custom Jinja2 filters
@app.template_filter('number_format')
def number_format_filter(value):
    """Format numbers with commas for better readability."""
    if value is None:
        return '0'
    try:
        return "{:,}".format(int(value))
    except (ValueError, TypeError):
        return str(value)

@app.template_filter('min')
def min_filter(value):
    """Return minimum value from iterable."""
    if not value:
        return 0
    try:
        return min(value)
    except (ValueError, TypeError):
        return 0

@app.template_filter('max')
def max_filter(value):
    """Return maximum value from iterable."""
    if not value:
        return 0
    try:
        return max(value)
    except (ValueError, TypeError):
        return 0

@app.template_filter('length')
def length_filter(value):
    """Return length of value."""
    if value is None:
        return 0
    try:
        return len(value)
    except (TypeError, AttributeError):
        return 0

# Global database inspector instance
db_inspector = None

def get_db_inspector():
    """Get or create database inspector instance."""
    global db_inspector
    if db_inspector is None:
        db_path = os.environ.get('DB_PATH', 'training_responses.db')
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        db_inspector = DatabaseInspector(db_path)
    return db_inspector

def safe_serialize(obj):
    """Safely serialize objects for JSON display."""
    if isinstance(obj, (list, dict)):
        return json.dumps(obj, indent=2, default=str)
    return str(obj)

@app.route('/')
def dashboard():
    """Dashboard page with database statistics."""
    try:
        inspector = get_db_inspector()
        stats = inspector.get_stats()
        successful_stats = inspector.get_successful_responses_stats()

        return render_template('dashboard.html',
                             stats=stats,
                             successful_stats=successful_stats)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/responses')
def responses():
    """List responses with pagination and filtering."""
    try:
        inspector = get_db_inspector()

        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        dataset = request.args.get('dataset', '')
        training_step = request.args.get('training_step', '')
        query_id = request.args.get('query_id', '')

        # Build filters
        filters = {}
        if dataset:
            filters['dataset'] = dataset
        if training_step:
            try:
                filters['training_step'] = int(training_step)
            except ValueError:
                pass
        if query_id:
            filters['query_id'] = query_id

        # Get filtered results
        all_responses = []
        for key, response in inspector.responses_db.items():
            match = True
            for filter_key, filter_value in filters.items():
                if response.get(filter_key) != filter_value:
                    match = False
                    break
            if match:
                all_responses.append({'key': key, 'data': response})

        # Paginate results
        total_responses = len(all_responses)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        responses_page = all_responses[start_idx:end_idx]

        # Get unique values for filters
        unique_datasets = inspector.get_unique_values('dataset')
        unique_training_steps = sorted(inspector.get_unique_values('training_step'))
        unique_query_ids = inspector.get_unique_values('query_id')

        total_pages = (total_responses + per_page - 1) // per_page

        return render_template('responses.html',
                             responses=responses_page,
                             page=page,
                             per_page=per_page,
                             total_pages=total_pages,
                             total_responses=total_responses,
                             filters=filters,
                             unique_datasets=unique_datasets,
                             unique_training_steps=unique_training_steps,
                             unique_query_ids=unique_query_ids)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/response/<key>')
def response_detail(key):
    """Show individual response details."""
    try:
        inspector = get_db_inspector()
        response = inspector.get_response_by_key(key)

        if not response:
            abort(404)

        # Get the associated query
        query_id = response.get('query_id')
        query = None
        if query_id:
            query_data = inspector.get_query_by_id(query_id)
            if query_data:
                query = query_data.get('query')

        return render_template('response_detail.html',
                             response_key=key,
                             response=response,
                             query=query,
                             safe_serialize=safe_serialize)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/queries')
def queries():
    """List queries with pagination."""
    try:
        inspector = get_db_inspector()

        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        has_success = request.args.get('has_success', '')

        # Get all queries
        all_queries = []
        for query_id, query_data in inspector.queries_db.items():
            successful_responses = query_data.get('successful_responses', [])
            if has_success == 'yes' and not successful_responses:
                continue
            elif has_success == 'no' and successful_responses:
                continue

            all_queries.append({
                'query_id': query_id,
                'data': query_data,
                'has_success': len(successful_responses) > 0,
                'success_count': len(successful_responses)
            })

        # Sort by success count (descending)
        all_queries.sort(key=lambda x: x['success_count'], reverse=True)

        # Paginate results
        total_queries = len(all_queries)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        queries_page = all_queries[start_idx:end_idx]

        total_pages = (total_queries + per_page - 1) // per_page

        return render_template('queries.html',
                             queries=queries_page,
                             page=page,
                             per_page=per_page,
                             total_pages=total_pages,
                             total_queries=total_queries,
                             has_success=has_success)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/query/<query_id>')
def query_detail(query_id):
    """Show individual query details."""
    try:
        inspector = get_db_inspector()
        query_data = inspector.get_query_by_id(query_id)

        if not query_data:
            abort(404)

        # Get successful responses for this query
        successful_responses = query_data.get('successful_responses', [])
        successful_response_details = []

        for resp in successful_responses:
            response_id = resp['response_id']
            response_data = inspector.get_response_by_key(response_id)
            if response_data:
                successful_response_details.append({
                    'response_id': response_id,
                    'response_length': resp.get('response_length', 0),
                    'response_data': response_data
                })

        return render_template('query_detail.html',
                             query_id=query_id,
                             query_data=query_data,
                             successful_responses=successful_response_details,
                             safe_serialize=safe_serialize)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/search')
def search():
    """Search interface."""
    try:
        inspector = get_db_inspector()

        query = request.args.get('q', '')
        search_type = request.args.get('type', 'responses')  # responses or queries
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        results = []
        total_results = 0

        if query and len(query) >= 3:
            if search_type == 'responses':
                # Search in response text
                for key, response in inspector.responses_db.items():
                    response_text = response.get('response_text', '').lower()
                    if query.lower() in response_text:
                        results.append({'key': key, 'data': response, 'type': 'response'})
                        total_results += 1
                        if total_results >= 1000:  # Limit search results
                            break
            elif search_type == 'queries':
                # Search in query text
                for query_id, query_data in inspector.queries_db.items():
                    query_text = query_data.get('query', '').lower()
                    if query.lower() in query_text:
                        results.append({'query_id': query_id, 'data': query_data, 'type': 'query'})
                        total_results += 1
                        if total_results >= 1000:  # Limit search results
                            break

        # Paginate results
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        results_page = results[start_idx:end_idx]
        total_pages = (total_results + per_page - 1) // per_page

        return render_template('search.html',
                             query=query,
                             search_type=search_type,
                             results=results_page,
                             page=page,
                             per_page=per_page,
                             total_pages=total_pages,
                             total_results=total_results)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/stats')
def api_stats():
    """API endpoint for database statistics."""
    try:
        inspector = get_db_inspector()
        stats = inspector.get_stats()
        successful_stats = inspector.get_successful_responses_stats()
        return jsonify({**stats, **successful_stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def teardown_db(exception):
    """Clean up database connections."""
    global db_inspector
    if db_inspector:
        db_inspector.close()
        db_inspector = None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)