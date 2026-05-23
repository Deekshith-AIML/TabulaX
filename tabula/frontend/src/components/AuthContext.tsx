import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import { apiUrl, SESSION_EXPIRED_EVENT } from '../lib/api';

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  register: (username: string, password: string, email: string, fullName: string) => Promise<boolean>;
}

interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  created_at?: string;
  last_login?: string;
  is_active: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Check if token exists in localStorage
    const storedToken = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user');
    
    if (storedToken && storedUser) {
      setUser(JSON.parse(storedUser));
      setIsAuthenticated(true);
      
      // Fetch user profile to validate token
      fetchUserProfile(storedToken);
    }
  }, []);

  // Any API call that hits a 401 dispatches SESSION_EXPIRED_EVENT (see
  // lib/api.ts). Clear the session so the UI returns to the login screen
  // instead of failing silently on subsequent actions.
  useEffect(() => {
    const onSessionExpired = () => {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setUser(null);
      setIsAuthenticated(false);
    };
    window.addEventListener(SESSION_EXPIRED_EVENT, onSessionExpired);
    return () => window.removeEventListener(SESSION_EXPIRED_EVENT, onSessionExpired);
  }, []);

  const fetchUserProfile = async (authToken: string) => {
    try {
      const response = await fetch(apiUrl('/users/me/'), {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });
      
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
      } else {
        // Token is invalid or expired
        logout();
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
      logout();
    }
  };

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await fetch(apiUrl('/token'), {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        const { access_token, user: userData } = data;
        
        // Store token and user data
        localStorage.setItem('token', access_token);
        localStorage.setItem('user', JSON.stringify(userData));
        
        setUser(userData);
        setIsAuthenticated(true);
        return true;
      } else {
        const errorData = await response.json();
        console.error('Login failed:', errorData.detail);
        return false;
      }
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const register = async (username: string, password: string, email: string, fullName: string): Promise<boolean> => {
    try {
      const response = await fetch(apiUrl('/register/'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username,
          password,
          email,
          full_name: fullName
        })
      });
      
      if (response.ok) {
        // Automatically log in after successful registration
        return await login(username, password);
      } else {
        const errorData = await response.json();
        console.error('Registration failed:', errorData.detail);
        return false;
      }
    } catch (error) {
      console.error('Registration error:', error);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, logout, register }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
