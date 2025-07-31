"""
Unit tests for API endpoints
"""

import pytest
import json
import io
from pathlib import Path
from unittest.mock import patch, Mock

from fastapi import status


@pytest.mark.unit
class TestUploadAPI:
    """Test upload API endpoints"""
    
    def test_upload_endpoint_success(self, client, sample_video_path):
        """Test successful video upload"""
        with open(sample_video_path, "rb") as video_file:
            response = client.post(
                "/api/upload/",
                files={"file": ("test_video.mp4", video_file, "video/mp4")}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "id" in data
        assert data["filename"] == "test_video.mp4"
        assert data["status"] == "success"
    
    def test_upload_unsupported_format(self, client, temp_dir):
        """Test upload with unsupported format"""
        # Create a fake .txt file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("This is not a video")
        
        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/upload/",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data["detail"]
    
    def test_upload_no_file(self, client):
        """Test upload without file"""
        response = client.post("/api/upload/")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_upload_status(self, client, sample_video_path):
        """Test getting upload status"""
        # First upload a video
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        # Get status
        response = client.get(f"/api/upload/status/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["id"] == video_id
        assert data["filename"] == "test.mp4"
        assert "upload_date" in data
    
    def test_get_upload_status_not_found(self, client):
        """Test getting status for non-existent video"""
        response = client.get("/api/upload/status/999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_video(self, client, sample_video_path):
        """Test deleting uploaded video"""
        # First upload a video
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        # Delete video
        response = client.delete(f"/api/upload/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Video deleted successfully"
        
        # Verify it's gone
        status_response = client.get(f"/api/upload/status/{video_id}")
        assert status_response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
class TestAnalysisAPI:
    """Test analysis API endpoints"""
    
    def test_start_analysis(self, client, sample_video_path):
        """Test starting video analysis"""
        # Upload video first
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        # Start analysis
        with patch('backend.api.routes.analysis.run_analysis_pipeline'):
            response = client.post(f"/api/analysis/start/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["video_id"] == video_id
        assert data["status"] == "started"
    
    def test_start_analysis_video_not_found(self, client):
        """Test starting analysis for non-existent video"""
        response = client.post("/api/analysis/start/999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_analysis_status(self, client, sample_video_path):
        """Test getting analysis status"""
        # Upload video first
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        # Get status
        response = client.get(f"/api/analysis/status/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["video_id"] == video_id
        assert "status" in data
        assert "progress" in data
    
    def test_cancel_analysis(self, client, sample_video_path):
        """Test cancelling analysis"""
        # Upload and start analysis
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        # Mock the video status as processing
        with patch('backend.database.crud.get_video') as mock_get_video:
            mock_video = Mock()
            mock_video.status = "processing"
            mock_get_video.return_value = mock_video
            
            with patch('backend.database.crud.update_video_status'):
                response = client.post(f"/api/analysis/cancel/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Analysis cancelled successfully"


@pytest.mark.unit
class TestResultsAPI:
    """Test results API endpoints"""
    
    def setup_analysis_data(self, client, sample_video_path):
        """Helper to setup video and analysis data"""
        # Upload video
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        # Mock analysis data
        with patch('backend.database.crud.get_video') as mock_get_video, \
             patch('backend.database.crud.get_analysis_by_video') as mock_get_analysis, \
             patch('backend.database.crud.get_player_scores') as mock_get_players, \
             patch('backend.database.crud.get_team_statistics') as mock_get_teams, \
             patch('backend.database.crud.get_key_moments') as mock_get_moments:
            
            mock_video = Mock()
            mock_video.id = video_id
            mock_get_video.return_value = mock_video
            
            mock_analysis = Mock()
            mock_analysis.id = 1
            mock_analysis.scores = {"overall": 85.5}
            mock_get_analysis.return_value = mock_analysis
            
            mock_get_players.return_value = []
            mock_get_teams.return_value = {}
            mock_get_moments.return_value = []
            
            return video_id
    
    def test_get_analysis_results(self, client, sample_video_path):
        """Test getting analysis results"""
        video_id = self.setup_analysis_data(client, sample_video_path)
        
        with patch('backend.database.crud.get_video') as mock_get_video, \
             patch('backend.database.crud.get_analysis_by_video') as mock_get_analysis, \
             patch('backend.database.crud.get_player_scores') as mock_get_players, \
             patch('backend.database.crud.get_team_statistics') as mock_get_teams, \
             patch('backend.database.crud.get_key_moments') as mock_get_moments:
            
            # Setup mocks
            mock_video = Mock()
            mock_get_video.return_value = mock_video
            
            mock_analysis = Mock()
            mock_analysis.id = 1
            mock_analysis.scores = {"overall": 85.5}
            mock_get_analysis.return_value = mock_analysis
            
            mock_get_players.return_value = []
            mock_get_teams.return_value = {}
            mock_get_moments.return_value = []
            
            response = client.get(f"/api/results/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["video_id"] == video_id
        assert data["analysis_id"] == 1
        assert "overall_scores" in data
        assert "player_scores" in data
        assert "team_statistics" in data
    
    def test_get_results_not_found(self, client):
        """Test getting results for non-existent video"""
        response = client.get("/api/results/999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_player_results(self, client, sample_video_path):
        """Test getting player-specific results"""
        video_id = self.setup_analysis_data(client, sample_video_path)
        
        with patch('backend.database.crud.get_analysis_by_video') as mock_get_analysis, \
             patch('backend.database.crud.get_player_scores') as mock_get_players:
            
            mock_analysis = Mock()
            mock_analysis.id = 1
            mock_get_analysis.return_value = mock_analysis
            
            mock_get_players.return_value = []
            
            response = client.get(f"/api/results/{video_id}/players")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["video_id"] == video_id
        assert "players" in data
    
    def test_get_timeline(self, client, sample_video_path):
        """Test getting timeline events"""
        video_id = self.setup_analysis_data(client, sample_video_path)
        
        with patch('backend.database.crud.get_analysis_by_video') as mock_get_analysis, \
             patch('backend.database.crud.get_timeline_events') as mock_get_events:
            
            mock_analysis = Mock()
            mock_analysis.id = 1
            mock_get_analysis.return_value = mock_analysis
            
            mock_get_events.return_value = []
            
            response = client.get(f"/api/results/{video_id}/timeline")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["video_id"] == video_id
        assert "timeline" in data


@pytest.mark.unit
class TestReportsAPI:
    """Test reports API endpoints"""
    
    def test_generate_report(self, client, sample_video_path):
        """Test generating analysis report"""
        # Upload video and mock analysis data
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        video_id = upload_response.json()["id"]
        
        with patch('backend.database.crud.get_video') as mock_get_video, \
             patch('backend.database.crud.get_analysis_by_video') as mock_get_analysis, \
             patch('backend.core.scoring.report_builder.ReportBuilder.generate_pdf_report') as mock_generate, \
             patch('backend.database.crud.create_report') as mock_create_report:
            
            # Setup mocks
            mock_video = Mock()
            mock_get_video.return_value = mock_video
            
            mock_analysis = Mock()
            mock_analysis.id = 1
            mock_get_analysis.return_value = mock_analysis
            
            mock_generate.return_value = Path("test_report.pdf")
            
            mock_report = Mock()
            mock_report.id = 1
            mock_create_report.return_value = mock_report
            
            response = client.post(f"/api/reports/generate/{video_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["report_id"] == 1
        assert data["type"] == "pdf"
        assert data["status"] == "generated"
    
    def test_get_report_templates(self, client):
        """Test getting available report templates"""
        response = client.get("/api/reports/templates")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "templates" in data
        assert len(data["templates"]) > 0
        
        # Check template structure
        template = data["templates"][0]
        assert "id" in template
        assert "name" in template
        assert "description" in template


@pytest.mark.unit
class TestHealthEndpoints:
    """Test health and system endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == "Football AI Analyzer API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "gpu_available" in data
        assert "models_loaded" in data


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API workflows"""
    
    def test_complete_analysis_workflow(self, client, sample_video_path):
        """Test complete workflow from upload to results"""
        # 1. Upload video
        with open(sample_video_path, "rb") as video_file:
            upload_response = client.post(
                "/api/upload/",
                files={"file": ("test.mp4", video_file, "video/mp4")}
            )
        
        assert upload_response.status_code == status.HTTP_200_OK
        video_id = upload_response.json()["id"]
        
        # 2. Check upload status
        status_response = client.get(f"/api/upload/status/{video_id}")
        assert status_response.status_code == status.HTTP_200_OK
        
        # 3. Start analysis (mocked)
        with patch('backend.api.routes.analysis.run_analysis_pipeline'):
            analysis_response = client.post(f"/api/analysis/start/{video_id}")
        
        assert analysis_response.status_code == status.HTTP_200_OK
        
        # 4. Check analysis status
        analysis_status_response = client.get(f"/api/analysis/status/{video_id}")
        assert analysis_status_response.status_code == status.HTTP_200_OK
    
    def test_error_handling(self, client):
        """Test API error handling"""
        # Test 404 errors
        response = client.get("/api/upload/status/999")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        response = client.get("/api/results/999")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test invalid endpoints
        response = client.get("/api/invalid")
        assert response.status_code == status.HTTP_404_NOT_FOUND