import React, { useState } from 'react';
import { X, Loader2, CheckCircle2, AlertCircle, User as UserIcon } from 'lucide-react';
import { updateProfile } from './authApi';

// Edit name / email / institute; the username is fixed. Saving persists
// locally and syncs to central tracking.
export default function ProfilePage({ user, onClose, onUpdated }) {
  const [form, setForm] = useState({
    name: user.name || '',
    email: user.email || '',
    institute: user.institute || 'personal',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);

  const update = (field) => (e) => {
    setForm((f) => ({ ...f, [field]: e.target.value }));
    setSaved(false);
    setError('');
  };

  const dirty =
    form.name !== (user.name || '') ||
    form.email !== (user.email || '') ||
    form.institute !== (user.institute || 'personal');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSaved(false);
    setLoading(true);
    try {
      const updated = await updateProfile(form);
      onUpdated(updated);
      setSaved(true);
    } catch (err) {
      setError(err.message || 'Could not save changes.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-gray-900/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      <div className="relative z-10 w-full max-w-md bg-white rounded-2xl shadow-2xl shadow-blue-500/10 border border-gray-100 animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
              <UserIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-800">Your profile</h2>
              <p className="text-xs text-gray-500">@{user.username}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 hover:bg-gray-100 p-2 rounded-lg transition-all"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} className="px-6 py-5 space-y-4">
          {saved && (
            <div className="flex items-start gap-2 p-3 rounded-xl bg-green-50 border border-green-200 text-green-700 text-sm">
              <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>Profile updated.</span>
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <ProfileField label="Full name" value={form.name} onChange={update('name')} required />
          <ProfileField
            label="Email"
            type="email"
            value={form.email}
            onChange={update('email')}
            required
          />
          <ProfileField
            label="Institute"
            value={form.institute}
            onChange={update('institute')}
            required
          />

          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-all"
            >
              Close
            </button>
            <button
              type="submit"
              disabled={loading || !dirty}
              className="flex items-center gap-2 px-5 py-2 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 text-white text-sm font-semibold shadow-lg shadow-blue-500/30 hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? 'Saving…' : 'Save changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function ProfileField({ label, type = 'text', ...props }) {
  return (
    <label className="block">
      <span className="block mb-1 text-sm font-medium text-gray-700">{label}</span>
      <input
        type={type}
        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 bg-white text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-400 transition-all duration-200"
        {...props}
      />
    </label>
  );
}
