from budget_parser_production import parse_budget_production
from proposal_parser_cloud import parse_proposal_cloud, parse_proposal_cloud_enhanced
from kpis_parser_cloud import kpis_parser_cloud
from firebase_service import FirebaseService


class DataPipeline:
    def __init__(self):
        self.firebase = FirebaseService()

    def process_project(self, project_id: str, files_dict: dict):
        """
        Process all files for a project and store in Firebase
        files_dict: {'budget': csv_content, 'proposal': text_content, 'kpis': csv_content}
        """
        results = {}

        try:
            # Parse budget
            if 'budget' in files_dict:
                budget_data = parse_budget_production(files_dict['budget'])
                self.firebase.store_parsed_data(project_id, 'budget', budget_data)
                results['budget'] = budget_data.get('status', 'error')

            # Parse proposal
            if 'proposal' in files_dict:
                proposal_data = parse_proposal_cloud_enhanced(files_dict['proposal'])
                self.firebase.store_parsed_data(project_id, 'proposal', proposal_data)
                results['proposal'] = 'success'

            # Parse KPIs
            if 'kpis' in files_dict:
                kpis_data = kpis_parser_cloud(files_dict['kpis'])
                self.firebase.store_parsed_data(project_id, 'kpis', kpis_data)
                results['kpis'] = kpis_data.get('status', 'error')

            # Create chunks for RAG
            chunk_count = self.firebase.create_chunks(project_id)
            results['chunks_created'] = chunk_count

            return {'status': 'success', 'results': results}

        except Exception as e:
            return {'status': 'error', 'error': str(e)}