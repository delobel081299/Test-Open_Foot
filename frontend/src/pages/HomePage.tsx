import React from 'react';
import { Link } from 'react-router-dom';
import { 
  CloudArrowUpIcon, 
  ChartBarIcon, 
  CpuChipIcon, 
  EyeIcon,
  PlayIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';

const HomePage: React.FC = () => {
  const features = [
    {
      name: 'Advanced Video Analysis',
      description: 'AI-powered analysis of football videos with player tracking, biomechanics, and tactical insights.',
      icon: EyeIcon,
      color: 'bg-blue-500'
    },
    {
      name: 'Real-time Processing',
      description: 'Fast GPU-accelerated processing with state-of-the-art models for maximum precision.',
      icon: CpuChipIcon,
      color: 'bg-green-500'
    },
    {
      name: 'Detailed Reports',
      description: 'Comprehensive analysis reports with actionable insights and improvement recommendations.',
      icon: DocumentTextIcon,
      color: 'bg-purple-500'
    },
    {
      name: 'Performance Metrics',
      description: 'Track technical skills, biomechanics, and tactical awareness with precise scoring.',
      icon: ChartBarIcon,
      color: 'bg-yellow-500'
    }
  ];

  const steps = [
    {
      step: '1',
      title: 'Upload Video',
      description: 'Upload your football match or training video',
      icon: CloudArrowUpIcon
    },
    {
      step: '2',
      title: 'AI Analysis',
      description: 'Our AI analyzes players, movements, and tactics',
      icon: CpuChipIcon
    },
    {
      step: '3',
      title: 'View Results',
      description: 'Get detailed insights and performance scores',
      icon: ChartBarIcon
    },
    {
      step: '4',
      title: 'Generate Report',
      description: 'Create comprehensive PDF reports',
      icon: DocumentTextIcon
    }
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <div className="text-center">
        <div className="mx-auto w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center mb-8">
          <span className="text-3xl">⚽</span>
        </div>
        <h1 className="text-4xl font-bold text-gray-900 sm:text-6xl">
          Football AI Analyzer
        </h1>
        <p className="mt-6 text-lg leading-8 text-gray-600 max-w-3xl mx-auto">
          Advanced AI-powered football video analysis platform. Analyze player performance, 
          biomechanics, and tactical decisions with cutting-edge computer vision and machine learning.
        </p>
        <div className="mt-10 flex items-center justify-center gap-x-6">
          <Link
            to="/upload"
            className="btn btn-primary btn-lg flex items-center space-x-2"
          >
            <CloudArrowUpIcon className="h-5 w-5" />
            <span>Start Analysis</span>
          </Link>
          <button className="btn btn-secondary btn-lg flex items-center space-x-2">
            <PlayIcon className="h-5 w-5" />
            <span>Watch Demo</span>
          </button>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16">
        <div className="text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Advanced Football Analysis
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Leverage the power of AI to analyze every aspect of football performance
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <div key={feature.name} className="card hover:shadow-xl transition-shadow">
                <div className="card-body text-center">
                  <div className={`mx-auto w-12 h-12 ${feature.color} rounded-lg flex items-center justify-center mb-4`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {feature.name}
                  </h3>
                  <p className="text-gray-600 text-sm">
                    {feature.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* How it Works */}
      <div className="py-16 bg-gray-50 -mx-8 px-8 rounded-lg">
        <div className="text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            How It Works
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Simple steps to get comprehensive football analysis
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-12 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <div key={step.step} className="relative">
                {/* Connection line */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-8 left-16 w-full h-0.5 bg-gray-300 z-0"></div>
                )}
                
                <div className="relative z-10 text-center">
                  <div className="mx-auto w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mb-4">
                    <Icon className="h-8 w-8 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                    <span className="text-white text-xs font-bold">{step.step}</span>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {step.title}
                  </h3>
                  <p className="text-gray-600 text-sm">
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Stats Section */}
      <div className="py-16">
        <div className="text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Trusted by Professionals
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Used by coaches, analysts, and players worldwide
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-8 sm:grid-cols-3">
          <div className="text-center">
            <div className="text-4xl font-bold text-blue-600">95%</div>
            <div className="text-sm text-gray-600 mt-2">Analysis Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-green-600">60 FPS</div>
            <div className="text-sm text-gray-600 mt-2">Processing Speed</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-purple-600">30+</div>
            <div className="text-sm text-gray-600 mt-2">Analysis Metrics</div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-blue-600 text-white py-16 -mx-8 px-8 rounded-lg text-center">
        <h2 className="text-3xl font-bold mb-4">
          Ready to Analyze Your Game?
        </h2>
        <p className="text-lg mb-8 text-blue-100">
          Upload your video and get instant AI-powered football analysis
        </p>
        <Link
          to="/upload"
          className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-blue-600 bg-white hover:bg-gray-50 transition-colors"
        >
          <CloudArrowUpIcon className="h-5 w-5 mr-2" />
          Get Started Now
        </Link>
      </div>
    </div>
  );
};

export default HomePage;