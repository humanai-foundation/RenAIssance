import React, { useCallback, useState } from 'react';
import { Upload, FileImage, FileText, X, Loader2, UploadCloud } from 'lucide-react';

export default function UploadZone({ onFilesSelected, isLoading }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  }, []);

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files);
    processFiles(files);
  }, []);

  const processFiles = (files) => {
    const validFiles = files.filter((file) => {
      const isPdf = file.type === 'application/pdf';
      const isImage = file.type.startsWith('image/');
      return isPdf || isImage;
    });

    if (validFiles.length > 0) {
      setSelectedFiles(validFiles);
      onFilesSelected(validFiles);
    }
  };

  const removeFile = (index) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    setSelectedFiles(newFiles);
    onFilesSelected(newFiles.length > 0 ? newFiles : null);
  };

  const getFileIcon = (file) => {
    if (file.type === 'application/pdf') {
      return <FileText className="w-8 h-8 text-red-500" />;
    }
    return <FileImage className="w-8 h-8 text-blue-500" />;
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-in">
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${isDragging
          ? 'border-blue-500 bg-blue-50/80 scale-[1.02] shadow-xl shadow-blue-500/30'
          : 'border-gray-300 bg-white/60 backdrop-blur-sm hover:border-blue-400 hover:bg-blue-50/30 hover:shadow-lg'
          } ${isLoading ? 'pointer-events-none opacity-60' : ''}`}
      >
        {/* Subtle background pattern */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none rounded-2xl overflow-hidden">
          <div className="absolute inset-0" style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, currentColor 1px, transparent 1px)`,
            backgroundSize: '24px 24px'
          }} />
        </div>

        {isLoading ? (
          <div className="flex flex-col items-center relative z-10">
            <div className="relative">
              <Loader2 className="w-16 h-16 text-blue-500 animate-spin" />
              <div className="absolute inset-0 blur-xl bg-blue-400/30 animate-pulse" />
            </div>
            <p className="mt-4 text-lg font-semibold text-blue-600">
              Processing files...
            </p>
            <p className="text-sm text-gray-500 mt-1">This may take a moment</p>
          </div>
        ) : (
          <div className="relative z-10">
            {/* Floating animated icon */}
            <div
              className={`mx-auto w-24 h-24 rounded-2xl flex items-center justify-center transition-all duration-500 ${isDragging
                ? 'bg-gradient-to-br from-blue-500 to-indigo-600 scale-110 shadow-xl shadow-blue-500/30'
                : 'bg-gradient-to-br from-blue-100 to-indigo-100'
                }`}
            >
              <UploadCloud
                className={`w-12 h-12 transition-all duration-300 ${isDragging ? 'text-white animate-bounce' : 'text-blue-500 animate-float'
                  }`}
              />
            </div>

            <h3 className="mt-6 text-xl font-bold text-gray-800">
              {isDragging ? 'Drop your files here!' : 'Drag & drop files here'}
            </h3>

            <p className="mt-2 text-gray-500">or</p>

            <label className="mt-4 inline-block">
              <input
                type="file"
                multiple
                accept=".pdf,image/*"
                onChange={handleFileInput}
                className="hidden"
              />
              <span className="btn-gradient inline-flex items-center gap-2 cursor-pointer">
                <Upload className="w-5 h-5" />
                Browse Files
              </span>
            </label>

            <p className="mt-6 text-sm text-gray-400 flex items-center justify-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              Supports PDF documents and images (PNG, JPG, TIFF)
            </p>
          </div>
        )}

        {/* Animated border on drag */}
        {isDragging && (
          <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-indigo-500 to-blue-500 opacity-20"
              style={{ animation: 'shimmer 1s linear infinite', backgroundSize: '200% 100%' }} />
          </div>
        )}
      </div>

      {/* Selected files list */}
      {selectedFiles.length > 0 && !isLoading && (
        <div className="mt-6 space-y-3 animate-slide-up">
          <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Selected Files ({selectedFiles.length})
          </h4>

          {selectedFiles.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-4 p-4 bg-white/80 backdrop-blur-sm rounded-xl border border-gray-100 shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200"
            >
              <div className="p-2 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg">
                {getFileIcon(file)}
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-gray-800 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-gray-400 mt-0.5">{formatFileSize(file.size)}</p>
              </div>

              <button
                onClick={() => removeFile(index)}
                className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all duration-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
