import React, { useState } from 'react';
import { useAuth } from './AuthContext';
import { Eye, EyeOff, Loader2 } from 'lucide-react';

interface LoginProps {
  onToggleForm: () => void;
}

const Login: React.FC<LoginProps> = ({ onToggleForm }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (!username || !password) {
      setError('Please enter both username and password');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // console.log('Attempting to login with username:', username); // Keep for debugging if needed
      const success = await login(username, password);
      
      if (!success) {
        setError('Invalid username or password. Please try again.');
      } else {
        // Navigation to app will be handled by AuthContext listener in App.tsx
      }
    } catch (err) {
      console.error('Login error:', err);
      setError('An error occurred during login. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white p-8 rounded-2xl shadow-2xl max-w-md w-full border border-neutral-200">
      <h2 className="text-3xl font-bold text-center text-neutral-800 mb-8">
        Welcome Back
      </h2>
      
      {error && (
        <div className="bg-red-50 border-l-4 border-red-600 text-red-700 p-4 rounded-md mb-6 text-sm" role="alert">
          <p>{error}</p>
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="login-username" className="block text-neutral-700 text-sm font-semibold mb-2">
            Username or Email
          </label>
          <input
            id="login-username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="input w-full py-2.5" // Use .input from index.css
            placeholder="e.g., yourname or name@example.com"
            required
            disabled={isLoading}
          />
        </div>
        
        <div>
          <label htmlFor="login-password" className="block text-neutral-700 text-sm font-semibold mb-2">
            Password
          </label>
          <div className="relative">
            <input
              id="login-password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="input w-full pr-10 py-2.5" // Use .input, add padding for icon
              placeholder="Enter your password"
              required
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 px-3 flex items-center text-neutral-500 hover:text-primary transition-colors rounded-r-md focus:outline-none"
              title={showPassword ? "Hide password" : "Show password"}
              disabled={isLoading}
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
        </div>
        
        <div>
          <button
            type="submit"
            disabled={isLoading}
            className="btn btn-primary w-full flex items-center justify-center py-2.5 text-base" // Use .btn .btn-primary
          >
            {isLoading ? (
              <>
                <Loader2 size={20} className="animate-spin mr-2" />
                Signing In...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </div>
        
        <div className="text-center pt-2">
          <p className="text-sm text-neutral-600">
            Don't have an account?{" "}
            <button
              type="button"
              onClick={onToggleForm}
              className="font-semibold text-primary hover:text-primary-dark focus:outline-none focus:underline transition-colors"
              disabled={isLoading}
            >
              Create an account
            </button>
          </p>
        </div>
      </form>
    </div>
  );
};

export default Login;
