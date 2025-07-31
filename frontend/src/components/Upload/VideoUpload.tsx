import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, XMarkIcon, PlayIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface VideoUploadProps {
  onUpload: (file: File) => void;
  isUploading?: boolean;
  progress?: number;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ 
  onUpload, 
  isUploading = false, 
  progress = 0 
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      // Validate file size (max 2GB)
      const maxSize = 2 * 1024 * 1024 * 1024; // 2GB in bytes
      if (file.size > maxSize) {
        toast.error('File size must be less than 2GB');
        return;
      }

      setSelectedFile(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    multiple: false,
    disabled: isUploading
  });

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  const handleRemove = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (selectedFile && !isUploading) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <PlayIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-medium text-gray-900">{selectedFile.name}</h3>
                <p className="text-sm text-gray-500">
                  {formatFileSize(selectedFile.size)} • {selectedFile.type}
                </p>
              </div>
            </div>
            <button
              onClick={handleRemove}
              className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>

          {previewUrl && (
            <div className="mb-4">
              <video
                src={previewUrl}
                controls
                className="w-full max-h-64 rounded-lg bg-black"
              >
                Your browser does not support the video tag.
              </video>
            </div>
          )}

          <div className="flex space-x-3">
            <button
              onClick={handleUpload}
              className="btn btn-primary flex-1"
            >
              Upload & Analyze
            </button>
            <button
              onClick={handleRemove}
              className="btn btn-secondary"
            >
              Remove
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (isUploading) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <CloudArrowUpIcon className="h-8 w-8 text-blue-600 animate-pulse" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Uploading Video...
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              {selectedFile?.name}
            </p>
            
            <div className="progress-bar mb-2">
              <div 
                className="progress-fill" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-500">
              {Math.round(progress)}% complete
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={`card cursor-pointer transition-all ${
        isDragActive 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-300 hover:border-gray-400'
      }`}
    >
      <input {...getInputProps()} />
      <div className="card-body text-center py-12">
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        
        {isDragActive ? (
          <div>
            <p className="text-lg font-medium text-blue-600 mb-2">
              Drop your video here
            </p>
            <p className="text-sm text-blue-500">
              Release to upload
            </p>
          </div>
        ) : (
          <div>
            <p className="text-lg font-medium text-gray-900 mb-2">
              Upload football video
            </p>
            <p className="text-sm text-gray-500 mb-4">
              Drag and drop your video file here, or click to browse
            </p>
          </div>
        )}

        <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-500 mt-4">
          <span className="px-2 py-1 bg-gray-100 rounded">MP4</span>
          <span className="px-2 py-1 bg-gray-100 rounded">AVI</span>
          <span className="px-2 py-1 bg-gray-100 rounded">MOV</span>
          <span className="px-2 py-1 bg-gray-100 rounded">MKV</span>
          <span className="px-2 py-1 bg-gray-100 rounded">WEBM</span>
        </div>
        
        <p className="text-xs text-gray-400 mt-2">
          Maximum file size: 2GB
        </p>
      </div>
    </div>
  );
};

export default VideoUpload;