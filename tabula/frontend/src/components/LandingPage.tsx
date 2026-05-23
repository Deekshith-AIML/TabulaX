import React, { useState } from 'react';
import { FileUp, Database, Code, Download, ChevronRight } from 'lucide-react';
import Login from './Login';
import Register from './Register';
import { useAuth } from './AuthContext';

const LandingPage: React.FC = () => {
  const [showLogin, setShowLogin] = useState(true);
  const { isAuthenticated } = useAuth();

  const toggleForm = () => {
    setShowLogin(!showLogin);
  };

  const features = [
    {
      icon: <FileUp className="h-10 w-10 text-primary" />,
      title: 'CSV Upload',
      description: 'Upload your CSV files and instantly view your data'
    },
    {
      icon: <Database className="h-10 w-10 text-primary" />,
      title: 'Database Connection',
      description: 'Connect directly to your database to transform data'
    },
    {
      icon: <Code className="h-10 w-10 text-primary" />,
      title: 'Smart Transformations',
      description: 'Define transformations by example and let AI do the work'
    },
    {
      icon: <Download className="h-10 w-10 text-primary" />,
      title: 'Easy Export',
      description: 'Download your transformed data in CSV format'
    }
  ];

  const testimonials = [
    {
      quote: "TabulaX has revolutionized how we handle data transformations. What used to take hours now takes minutes.",
      author: "Sarah J., Data Analyst"
    },
    {
      quote: "The ability to define transformations by example is incredibly intuitive. It's like having a data scientist on demand.",
      author: "Michael T., Business Intelligence Manager"
    },
    {
      quote: "I'm not a programmer, but TabulaX makes me feel like one. It's empowering to transform data without writing code.",
      author: "Elena R., Marketing Director"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:w-full lg:pb-28 xl:pb-32">
            <div className="pt-6 px-4 sm:px-6 lg:px-8">
              <nav className="relative flex items-center justify-between sm:h-10">
                <div className="flex items-center flex-grow flex-shrink-0 lg:flex-grow-0">
                  <div className="flex items-center justify-between w-full md:w-auto">
                    <a href="/" className="flex">
                      <span className="sr-only">TabulaX</span>
                      <h1 className="text-3xl font-bold text-primary">TabulaX</h1>
                    </a>
                  </div>
                </div>
                <div className="hidden md:block md:ml-10 md:pr-4 md:space-x-8">
                  <a href="#features" className="font-medium text-gray-500 hover:text-gray-900">Features</a>
                  <a href="#how-it-works" className="font-medium text-gray-500 hover:text-gray-900">How it works</a>
                  <a href="#testimonials" className="font-medium text-gray-500 hover:text-gray-900">Testimonials</a>
                  {isAuthenticated ? (
                    <a href="/app" className="font-medium text-primary hover:text-blue-700">Dashboard</a>
                  ) : (
                    <button 
                      onClick={() => setShowLogin(true)}
                      className="font-medium text-primary hover:text-blue-700"
                    >
                      Sign in
                    </button>
                  )}
                </div>
              </nav>
            </div>

            <div className="flex flex-col md:flex-row mt-10 mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              <div className="md:w-1/2 lg:pr-8 xl:pr-16 md:pt-16 flex flex-col justify-center">
                <h2 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span className="block">Transform your data</span>
                  <span className="block text-primary">by example</span>
                </h2>
                <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                  TabulaX helps you transform tabular data without writing complex code. Simply provide examples of the transformation you want, and let our AI do the rest.
                </p>
                <div className="mt-5 sm:mt-8 sm:flex sm:justify-start">
                  {isAuthenticated ? (
                    <div className="rounded-md shadow">
                      <a
                        href="/app"
                        className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary hover:bg-blue-700 md:py-4 md:text-lg md:px-10"
                      >
                        Go to Dashboard
                      </a>
                    </div>
                  ) : (
                    <div className="rounded-md shadow">
                      <button
                        onClick={() => setShowLogin(false)}
                        className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary hover:bg-blue-700 md:py-4 md:text-lg md:px-10"
                      >
                        Get Started
                      </button>
                    </div>
                  )}
                  <div className="mt-3 sm:mt-0 sm:ml-3">
                    <a
                      href="#how-it-works"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-primary bg-blue-100 hover:bg-blue-200 md:py-4 md:text-lg md:px-10"
                    >
                      Learn More
                    </a>
                  </div>
                </div>
              </div>
              
              {!isAuthenticated && (
                <div className="md:w-1/2 mt-10 md:mt-0 flex justify-center items-center">
                  {showLogin ? (
                    <Login onToggleForm={toggleForm} />
                  ) : (
                    <Register onToggleForm={toggleForm} />
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-primary font-semibold tracking-wide uppercase">Features</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Everything you need for data transformation
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              TabulaX provides a complete set of tools to make data transformation easy and intuitive.
            </p>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
              {features.map((feature, index) => (
                <div key={index} className="pt-6">
                  <div className="flow-root bg-gray-50 rounded-lg px-6 pb-8">
                    <div className="-mt-6">
                      <div>
                        <span className="inline-flex items-center justify-center p-3 bg-blue-50 rounded-md shadow-lg">
                          {feature.icon}
                        </span>
                      </div>
                      <h3 className="mt-8 text-lg font-medium text-gray-900 tracking-tight">{feature.title}</h3>
                      <p className="mt-5 text-base text-gray-500">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* How it Works Section */}
      <div id="how-it-works" className="py-16 bg-gray-50 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-primary font-semibold tracking-wide uppercase">How it Works</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Transform data in three simple steps
            </p>
          </div>

          <div className="mt-10">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-white">
                  <span className="text-lg font-bold">1</span>
                </div>
                <div className="ml-16">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Upload your data</h3>
                  <p className="mt-2 text-base text-gray-500">
                    Upload a CSV file or connect to your database to import the data you want to transform.
                  </p>
                </div>
              </div>

              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-white">
                  <span className="text-lg font-bold">2</span>
                </div>
                <div className="ml-16">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Provide examples</h3>
                  <p className="mt-2 text-base text-gray-500">
                    Select a column and provide a few examples of how you want to transform the data.
                  </p>
                </div>
              </div>

              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-white">
                  <span className="text-lg font-bold">3</span>
                </div>
                <div className="ml-16">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Apply & download</h3>
                  <p className="mt-2 text-base text-gray-500">
                    Our AI will generate and apply the transformation to your entire dataset. Download the results.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials Section */}
      <div id="testimonials" className="py-16 bg-white overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center mb-10">
            <h2 className="text-base text-primary font-semibold tracking-wide uppercase">Testimonials</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              What our users say
            </p>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="bg-blue-50 rounded-lg p-6 shadow-sm">
                <div className="text-lg text-gray-600 italic mb-4">"{testimonial.quote}"</div>
                <div className="font-medium text-gray-900">{testimonial.author}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-primary">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:py-16 lg:px-8 lg:flex lg:items-center lg:justify-between">
          <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            <span className="block">Ready to transform your data?</span>
            <span className="block text-blue-100">Start using TabulaX today.</span>
          </h2>
          <div className="mt-8 flex lg:mt-0 lg:flex-shrink-0">
            <div className="inline-flex rounded-md shadow">
              {isAuthenticated ? (
                <a
                  href="/app"
                  className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-primary bg-white hover:bg-blue-50"
                >
                  Go to Dashboard
                  <ChevronRight className="ml-2 h-5 w-5" />
                </a>
              ) : (
                <button
                  onClick={() => setShowLogin(false)}
                  className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-primary bg-white hover:bg-blue-50"
                >
                  Get Started
                  <ChevronRight className="ml-2 h-5 w-5" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-white">TabulaX</h2>
              <p className="text-gray-400 mt-2">Transform your data by example</p>
            </div>
            <div className="flex space-x-6">
              <a href="#" className="text-gray-400 hover:text-gray-300">
                Terms
              </a>
              <a href="#" className="text-gray-400 hover:text-gray-300">
                Privacy
              </a>
              <a href="#" className="text-gray-400 hover:text-gray-300">
                Contact
              </a>
            </div>
          </div>
          <div className="mt-8 border-t border-gray-700 pt-8">
            <p className="text-gray-400 text-sm text-center">
              &copy; {new Date().getFullYear()} TabulaX. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
