import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { getAnalysisStatus, startAnalysis, cancelAnalysis } from '../services/api';

const AnalysisPage: React.FC = () => {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();
  const [isStarting, setIsStarting] = useState(false);

  const { data: status, isLoading, error, refetch } = useQuery(
    ['analysisStatus', videoId],
    () => getAnalysisStatus(Number(videoId)),
    {
      enabled: Boolean(videoId),
      refetchInterval: (data) => {
        // Refetch every 2 seconds if processing
        return data?.status === 'processing' ? 2000 : false;
      },
    }
  );

  useEffect(() => {
    if (status?.status === 'completed' && status.analysis_id) {
      toast.success('Analysis completed successfully!');
      setTimeout(() => {
        navigate(`/results/${videoId}`);
      }, 2000);
    }
  }, [status, videoId, navigate]);

  const handleStartAnalysis = async () => {
    if (!videoId) return;

    setIsStarting(true);
    try {
      await startAnalysis(Number(videoId));
      toast.success('Analysis started successfully!');
      refetch();
    } catch (error) {
      console.error('Failed to start analysis:', error);
      toast.error('Failed to start analysis. Please try again.');
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancelAnalysis = async () => {
    if (!videoId) return;

    try {
      await cancelAnalysis(Number(videoId));
      toast.success('Analysis cancelled');
      refetch();
    } catch (error) {
      console.error('Failed to cancel analysis:', error);
      toast.error('Failed to cancel analysis');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analysis status...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body text-center py-12">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Error Loading Analysis
          </h3>
          <p className="text-gray-600 mb-6">
            Failed to load analysis status. Please try again.
          </p>
          <button
            onClick={() => refetch()}
            className="btn btn-primary"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600';
      case 'processing': return 'text-blue-600';
      case 'failed': return 'text-red-600';
      case 'cancelled': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-16 w-16 text-green-500" />;
      case 'processing':
        return (
          <div className="relative">
            <div className="loading-spinner h-16 w-16 border-4"></div>
          </div>
        );
      case 'failed':
        return <ExclamationTriangleIcon className="h-16 w-16 text-red-500" />;
      default:
        return <PlayIcon className="h-16 w-16 text-gray-400" />;
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Video Analysis
        </h1>
        <p className="text-lg text-gray-600">
          AI-powered football analysis in progress
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Status Section */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-body text-center py-12">
              <div className="mb-6">
                {getStatusIcon(status?.status || 'pending')}
              </div>

              <h2 className={`text-2xl font-bold mb-2 ${getStatusColor(status?.status || 'pending')}`}>
                {status?.status === 'completed' && 'Analysis Complete!'}
                {status?.status === 'processing' && 'Analyzing Video...'}
                {status?.status === 'failed' && 'Analysis Failed'}
                {status?.status === 'cancelled' && 'Analysis Cancelled'}
                {!status?.status && 'Ready to Analyze'}
              </h2>

              <p className="text-gray-600 mb-8">
                {status?.status === 'completed' && 'Your video analysis is ready. Redirecting to results...'}
                {status?.status === 'processing' && 'AI models are analyzing your video. This may take a few minutes.'}
                {status?.status === 'failed' && 'Something went wrong during analysis. Please try again.'}
                {status?.status === 'cancelled' && 'Analysis was cancelled. You can restart if needed.'}
                {!status?.status && 'Click the button below to start analyzing your video.'}
              </p>

              {status?.status === 'processing' && (
                <div className="mb-6">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${status.progress || 25}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-500 mt-2">
                    {Math.round(status.progress || 25)}% complete
                  </p>
                </div>
              )}

              <div className="flex justify-center space-x-4">
                {!status?.status || status.status === 'cancelled' || status.status === 'failed' ? (
                  <button
                    onClick={handleStartAnalysis}
                    disabled={isStarting}
                    className="btn btn-primary flex items-center space-x-2"
                  >
                    <PlayIcon className="h-5 w-5" />
                    <span>{isStarting ? 'Starting...' : 'Start Analysis'}</span>
                  </button>
                ) : null}

                {status?.status === 'processing' && (
                  <button
                    onClick={handleCancelAnalysis}
                    className="btn btn-danger flex items-center space-x-2"
                  >
                    <StopIcon className="h-5 w-5" />
                    <span>Cancel</span>
                  </button>
                )}

                {status?.status === 'completed' && (
                  <button
                    onClick={() => navigate(`/results/${videoId}`)}
                    className="btn btn-success flex items-center space-x-2"
                  >
                    <CheckCircleIcon className="h-5 w-5" />
                    <span>View Results</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Info Section */}
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                Analysis Details
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Video ID:</span>
                  <span className="font-medium">#{videoId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Status:</span>
                  <span className={`font-medium capitalize ${getStatusColor(status?.status || 'pending')}`}>
                    {status?.status || 'Pending'}
                  </span>
                </div>
                {status?.analysis_id && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">Analysis ID:</span>
                    <span className="font-medium">#{status.analysis_id}</span>
                  </div>
                )}
                {status?.completed_at && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">Completed:</span>
                    <span className="font-medium">
                      {new Date(status.completed_at).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                Analysis Pipeline
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-3">
                {[
                  'Video preprocessing',
                  'Player detection',
                  'Movement tracking',
                  'Biomechanics analysis',
                  'Technical assessment',
                  'Tactical evaluation',
                  'Score calculation',
                  'Report generation'
                ].map((step, index) => (
                  <div key={step} className="flex items-center text-sm">
                    <div className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center ${
                      status?.status === 'completed' || 
                      (status?.status === 'processing' && index < 4)
                        ? 'bg-green-500' 
                        : 'bg-gray-300'
                    }`}>
                      {(status?.status === 'completed' || 
                        (status?.status === 'processing' && index < 4)) && (
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                      )}
                    </div>
                    <span className={
                      status?.status === 'completed' || 
                      (status?.status === 'processing' && index < 4)
                        ? 'text-gray-900' 
                        : 'text-gray-500'
                    }>
                      {step}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card bg-blue-50 border-blue-200">
            <div className="card-body">
              <h4 className="font-medium text-blue-900 mb-2">
                ⏱️ Processing Time
              </h4>
              <p className="text-sm text-blue-700">
                Analysis typically takes 2-5 minutes depending on video length 
                and complexity. You can safely close this page and return later.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;