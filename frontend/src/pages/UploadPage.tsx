import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

import VideoUpload from '../components/Upload/VideoUpload';
import { uploadVideo } from '../services/api';

const UploadPage: React.FC = () => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 10;
        });
      }, 500);

      const response = await uploadVideo(file, (progress) => {
        setUploadProgress(progress);
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      toast.success('Video uploaded successfully!');
      
      // Navigate to analysis page
      setTimeout(() => {
        navigate(`/analysis/${response.id}`);
      }, 1000);

    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('Upload failed. Please try again.');
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Upload Football Video
        </h1>
        <p className="text-lg text-gray-600">
          Upload your football match or training video to start AI-powered analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Section */}
        <div className="lg:col-span-2">
          <VideoUpload
            onUpload={handleUpload}
            isUploading={isUploading}
            progress={uploadProgress}
          />
        </div>

        {/* Info Section */}
        <div className="space-y-6">
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                Supported Formats
              </h3>
            </div>
            <div className="card-body">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
                  MP4 (recommended)
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
                  AVI
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
                  MOV
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
                  MKV
                </li>
                <li className="flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
                  WEBM
                </li>
              </ul>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                Requirements
              </h3>
            </div>
            <div className="card-body">
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• Minimum resolution: 640x480</li>
                <li>• Minimum frame rate: 15 FPS</li>
                <li>• Duration: 5 seconds - 1 hour</li>
                <li>• Maximum file size: 2GB</li>
                <li>• Clear view of players</li>
              </ul>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                Analysis Features
              </h3>
            </div>
            <div className="card-body">
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• Player detection & tracking</li>
                <li>• Biomechanical analysis</li>
                <li>• Technical skill assessment</li>
                <li>• Tactical movement analysis</li>
                <li>• Performance scoring</li>
                <li>• Detailed PDF reports</li>
              </ul>
            </div>
          </div>

          <div className="card bg-blue-50 border-blue-200">
            <div className="card-body">
              <h4 className="font-medium text-blue-900 mb-2">
                💡 Pro Tip
              </h4>
              <p className="text-sm text-blue-700">
                For best results, use videos with a side view of the field 
                and good lighting conditions. Multiple camera angles can 
                provide more comprehensive analysis.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;