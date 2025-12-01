import firebase_admin
from firebase_admin import credentials, firestore, storage
import json
from datetime import datetime


class FirebaseService:
    def __init__(self):
        # Initialize Firebase (use service account key)
        cred = credentials.Certificate('path/to/service-account-key.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'fluyx-yh-kb.appspot.com'
        })
        self.db = firestore.client()
        self.bucket = storage.bucket()

    def store_parsed_data(self, project_id: str, parser_type: str, data: dict):
        """Store parsed data from your parsers"""
        doc_ref = self.db.collection('projects').document(project_id)

        # Merge with existing data
        doc_ref.set({
            f'{parser_type}_data': data,
            'last_updated': datetime.now(),
            'parser_versions': {
                parser_type: 'production_gemini'
            }
        }, merge=True)

        print(f"Stored {parser_type} data for project {project_id}")

    def create_chunks(self, project_id: str, chunk_size: int = 1000):
        """Create text chunks from parsed data for RAG"""
        project_ref = self.db.collection('projects').document(project_id)
        project_data = project_ref.get().to_dict()

        chunks = []

        # Combine all text data from different parsers
        combined_text = self._combine_project_text(project_data)

        # Split into chunks
        for i in range(0, len(combined_text), chunk_size):
            chunk = {
                'project_id': project_id,
                'content': combined_text[i:i + chunk_size],
                'chunk_index': i // chunk_size,
                'metadata': {
                    'source_parsers': list(project_data.keys()),
                    'char_length': len(combined_text[i:i + chunk_size])
                },
                'created_at': datetime.now()
            }
            chunks.append(chunk)

        # Store chunks
        for chunk in chunks:
            self.db.collection('chunks').add(chunk)

        return len(chunks)

    def _combine_project_text(self, project_data: dict) -> str:
        """Combine text from all parsers into one searchable string"""
        text_parts = []

        # Budget data
        if 'budget_data' in project_data:
            budget = project_data['budget_data']
            text_parts.append(f"Budget: {budget.get('metadata', {}).get('project_title', '')}")
            for item in budget.get('budget_items', []):
                text_parts.append(f"Budget Item: {item.get('description', '')} - {item.get('requested_amount', '')}")

        # Proposal data
        if 'proposal_data' in project_data:
            proposal = project_data['proposal_data']
            text_parts.append(f"Proposal: {proposal.get('project_title', '')}")
            text_parts.append(f"Goal: {proposal.get('project_goal', '')}")
            text_parts.append(f"Description: {proposal.get('project_description', '')}")

        # KPIs data
        if 'kpis_data' in project_data:
            kpis = project_data['kpis_data']
            for milestone in kpis.get('milestones', []):
                for kpi in milestone.get('kpis', []):
                    text_parts.append(f"KPI: {kpi.get('kpi_description', '')}")

        return "\n".join(text_parts)

    def store_file(self, project_id: str, file_name: str, file_content: bytes):
        """Store original file in Firebase Storage"""
        blob = self.bucket.blob(f"projects/{project_id}/{file_name}")
        blob.upload_from_string(file_content)
        return blob.public_url

    def search_chunks(self, query: str, impact_area: str = None, limit: int = 10):
        """Search through project chunks (basic text search)"""
        chunks_ref = self.db.collection('chunks')

        if impact_area:
            chunks_ref = chunks_ref.where('metadata.impact_area', '==', impact_area)

        # This is a simple text search - for production, use vector search
        results = []
        for doc in chunks_ref.stream():
            chunk_data = doc.to_dict()
            if query.lower() in chunk_data['content'].lower():
                results.append({
                    'chunk_id': doc.id,
                    'content': chunk_data['content'],
                    'project_id': chunk_data['project_id'],
                    'score': chunk_data['content'].lower().count(query.lower())
                })

        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]