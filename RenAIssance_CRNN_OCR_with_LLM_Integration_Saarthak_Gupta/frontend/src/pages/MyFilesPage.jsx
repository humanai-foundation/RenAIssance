import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Archive,
  Calendar,
  Database,
  Download,
  Eye,
  FileText,
  Image as ImageIcon,
  Loader2,
  RefreshCw,
  Trash2,
  X,
} from 'lucide-react';
import {
  deleteStoredItem,
  downloadBlob,
  downloadStoredItem,
  fetchStorageOverview,
  fetchDatasetDetail,
  fetchTranscriptDetail,
} from '../services/storageApi';

function formatDate(value) {
  if (!value) return 'Unknown date';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function TagChips({ tags }) {
  if (!tags || tags.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-2">
      {tags.map((tag) => (
        <span key={tag} className="inline-block px-2 py-0.5 rounded-full text-[10px] font-medium bg-gray-100 text-gray-600 border border-gray-200">
          {tag}
        </span>
      ))}
    </div>
  );
}

function EmptyState({ title, description, color }) {
  return (
    <div className={`rounded-2xl border border-dashed p-8 text-center ${color}`}>
      <p className="font-semibold text-lg">{title}</p>
      <p className="text-sm mt-1 opacity-80">{description}</p>
    </div>
  );
}

function computeDynamicPreviewHeight(aspectRatio) {
  const safeRatio = Number.isFinite(aspectRatio) && aspectRatio > 0 ? aspectRatio : 1.5;
  const estimated = Math.round(460 / safeRatio);
  return Math.max(280, Math.min(620, estimated));
}

export default function MyFilesPage({ onBack }) {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [transcripts, setTranscripts] = useState([]);
  const [datasets, setDatasets] = useState([]);

  const [viewingTranscript, setViewingTranscript] = useState(null);
  const [viewingDataset, setViewingDataset] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [transcriptImageRatios, setTranscriptImageRatios] = useState({});
  const [datasetImageRatios, setDatasetImageRatios] = useState({});

  const totalItems = useMemo(() => transcripts.length + datasets.length, [transcripts, datasets]);

  const loadData = async (isRefresh = false) => {
    setError('');
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    try {
      const data = await fetchStorageOverview();
      setTranscripts(data.transcripts || []);
      setDatasets(data.datasets || []);
    } catch (err) {
      setError(err.message || 'Failed to load saved files');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleDelete = async (kind, id) => {
    const ok = window.confirm(`Delete ${id}? This cannot be undone.`);
    if (!ok) return;

    try {
      await deleteStoredItem(kind, id);
      await loadData(true);
      if (viewingTranscript?.metadata?.id === id) {
        setViewingTranscript(null);
      }
      if (viewingDataset?.metadata?.id === id) {
        setViewingDataset(null);
      }
    } catch (err) {
      setError(err.message || 'Delete failed');
    }
  };

  const handleDownload = async (kind, id) => {
    try {
      const blob = await downloadStoredItem(kind, id);
      downloadBlob(blob, `${id}.zip`);
    } catch (err) {
      setError(err.message || 'Download failed');
    }
  };

  const handleViewTranscript = async (id) => {
    setDetailLoading(true);
    setError('');
    setTranscriptImageRatios({});
    try {
      const detail = await fetchTranscriptDetail(id);
      setViewingTranscript(detail);
    } catch (err) {
      setError(err.message || 'Failed to load transcript');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleViewDataset = async (id) => {
    setDetailLoading(true);
    setError('');
    setDatasetImageRatios({});
    try {
      const detail = await fetchDatasetDetail(id);
      setViewingDataset(detail);
    } catch (err) {
      setError(err.message || 'Failed to load dataset details');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleTranscriptImageLoad = useCallback((key, event) => {
    const img = event.currentTarget;
    const ratio = img.naturalWidth > 0 && img.naturalHeight > 0
      ? img.naturalWidth / img.naturalHeight
      : 1.5;
    setTranscriptImageRatios((prev) => ({ ...prev, [key]: ratio }));
  }, []);

  const handleDatasetImageLoad = useCallback((key, event) => {
    const img = event.currentTarget;
    const ratio = img.naturalWidth > 0 && img.naturalHeight > 0
      ? img.naturalWidth / img.naturalHeight
      : 1.5;
    setDatasetImageRatios((prev) => ({ ...prev, [key]: ratio }));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-cyan-50">
      <header className="bg-white/90 backdrop-blur-md border-b border-gray-100 shadow-sm sticky top-0 z-30">
        <div className="w-full px-4 md:px-8 xl:px-12 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 text-white flex items-center justify-center shadow-md">
              <Archive className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">My Files</h1>
              <p className="text-xs text-gray-500">Persistent OCR outputs and datasets</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => loadData(true)}
              className="px-3 py-2 rounded-lg text-sm font-medium text-gray-600 hover:text-cyan-700 hover:bg-cyan-50 transition-colors"
              disabled={refreshing}
            >
              {refreshing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            </button>
            <button
              onClick={onBack}
              className="px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:text-cyan-700 hover:bg-cyan-50 transition-colors"
            >
              Home
            </button>
          </div>
        </div>
      </header>

      <main className="w-full px-4 md:px-8 xl:px-12 py-6 space-y-6">
        <div className="bg-white border border-gray-100 rounded-2xl shadow-sm px-4 py-3 text-sm text-gray-600">
          Stored items: <span className="font-semibold text-gray-800">{totalItems}</span>
        </div>

        {error && (
          <div className="p-3 rounded-xl border border-red-200 bg-red-50 text-red-700 text-sm">{error}</div>
        )}

        {loading ? (
          <div className="h-64 flex items-center justify-center text-gray-500">
            <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading files...
          </div>
        ) : (
          <div className="grid xl:grid-cols-2 gap-6">
            <section className="bg-white rounded-2xl border border-blue-100 shadow-sm overflow-hidden min-h-[72vh]">
              <div className="px-5 py-4 border-b border-blue-100 bg-gradient-to-r from-blue-50 to-white flex items-center justify-between">
                <h2 className="font-bold text-blue-800 flex items-center gap-2">
                  <FileText className="w-5 h-5" /> Transcriptions
                </h2>
                <span className="text-xs font-semibold text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                  {transcripts.length}
                </span>
              </div>

              <div className="p-4 space-y-3 h-[calc(72vh-4rem)] overflow-y-auto">
                {transcripts.length === 0 && (
                  <EmptyState
                    title="No saved transcriptions"
                    description="Completed OCR sessions will appear here."
                    color="text-blue-700 bg-blue-50/50"
                  />
                )}

                {transcripts.map((item) => (
                  <article key={item.id} className="border border-blue-100 rounded-xl p-4 bg-blue-50/40">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="font-semibold text-gray-800">{item.name || item.id}</p>
                        <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                          <Calendar className="w-3 h-3" /> {formatDate(item.created_at)}
                        </p>
                        <p className="text-xs text-blue-700 mt-1">{item.num_pages || item.num_files || 0} pages</p>
                        <TagChips tags={item.tags} />
                      </div>
                    </div>

                    <div className="mt-3 flex items-center gap-2">
                      <button
                        onClick={() => handleViewTranscript(item.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 transition-colors flex items-center gap-1"
                        disabled={detailLoading}
                      >
                        {detailLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Eye className="w-3.5 h-3.5" />} View
                      </button>
                      <button
                        onClick={() => handleDownload('transcripts', item.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-blue-700 border border-blue-200 hover:bg-blue-50 transition-colors flex items-center gap-1"
                      >
                        <Download className="w-3.5 h-3.5" /> Download
                      </button>
                      <button
                        onClick={() => handleDelete('transcripts', item.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-red-600 border border-red-200 hover:bg-red-50 transition-colors flex items-center gap-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" /> Delete
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="bg-white rounded-2xl border border-emerald-100 shadow-sm overflow-hidden min-h-[72vh]">
              <div className="px-5 py-4 border-b border-emerald-100 bg-gradient-to-r from-emerald-50 to-white flex items-center justify-between">
                <h2 className="font-bold text-emerald-800 flex items-center gap-2">
                  <Database className="w-5 h-5" /> Datasets
                </h2>
                <span className="text-xs font-semibold text-emerald-700 bg-emerald-100 px-2 py-1 rounded-full">
                  {datasets.length}
                </span>
              </div>

              <div className="p-4 space-y-3 h-[calc(72vh-4rem)] overflow-y-auto">
                {datasets.length === 0 && (
                  <EmptyState
                    title="No saved datasets"
                    description="Generated datasets are persisted here automatically."
                    color="text-emerald-700 bg-emerald-50/50"
                  />
                )}

                {datasets.map((item) => (
                  <article key={item.id} className="border border-emerald-100 rounded-xl p-4 bg-emerald-50/40">
                    <p className="font-semibold text-gray-800">{item.book_name || item.id}</p>
                    <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                      <Calendar className="w-3 h-3" /> {formatDate(item.created_at)}
                    </p>
                    <p className="text-xs text-emerald-700 mt-1">
                      {item.dataset_type || item.mode} • {item.num_samples || item.num_files || 0} samples • {item.format || 'unknown format'}
                    </p>
                    <TagChips tags={item.tags} />

                    <div className="mt-3 flex items-center gap-2">
                      <button
                        onClick={() => handleViewDataset(item.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 text-white hover:bg-emerald-700 transition-colors flex items-center gap-1"
                        disabled={detailLoading}
                      >
                        {detailLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Eye className="w-3.5 h-3.5" />} View
                      </button>
                      <button
                        onClick={() => handleDownload('datasets', item.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-emerald-700 border border-emerald-200 hover:bg-emerald-50 transition-colors flex items-center gap-1"
                      >
                        <Download className="w-3.5 h-3.5" /> Download
                      </button>
                      <button
                        onClick={() => handleDelete('datasets', item.id)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white text-red-600 border border-red-200 hover:bg-red-50 transition-colors flex items-center gap-1"
                      >
                        <Trash2 className="w-3.5 h-3.5" /> Delete
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          </div>
        )}
      </main>

      {viewingTranscript && (
        <div className="fixed inset-0 z-50 bg-gray-900/45 backdrop-blur-sm p-4 flex items-center justify-center">
          <div className="w-full max-w-4xl max-h-[85vh] bg-white rounded-2xl shadow-2xl border border-gray-100 flex flex-col overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
              <div>
                <p className="font-semibold text-gray-800">{viewingTranscript.metadata?.name || viewingTranscript.metadata?.id}</p>
                <p className="text-xs text-gray-500">{formatDate(viewingTranscript.metadata?.created_at)}</p>
                <TagChips tags={viewingTranscript.metadata?.tags} />
              </div>
              <button
                onClick={() => setViewingTranscript(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 overflow-y-auto space-y-3">
              <div className="grid md:grid-cols-3 gap-2 text-xs text-gray-600 bg-gray-50 rounded-xl p-3 border border-gray-100">
                <p><span className="font-semibold text-gray-700">Mode:</span> {viewingTranscript.metadata?.mode || 'n/a'}</p>
                <p><span className="font-semibold text-gray-700">Pages:</span> {viewingTranscript.metadata?.num_pages || viewingTranscript.metadata?.num_files || 0}</p>
                <p><span className="font-semibold text-gray-700">Source:</span> {viewingTranscript.metadata?.source || 'n/a'}</p>
                {viewingTranscript.metadata?.model_info?.preprocessing?.length > 0 && (
                  <p className="md:col-span-2"><span className="font-semibold text-gray-700">Preprocessing:</span> {viewingTranscript.metadata.model_info.preprocessing.join(', ')}</p>
                )}
                {viewingTranscript.metadata?.model_info?.detection_model && (
                  <p><span className="font-semibold text-gray-700">Detection:</span> {viewingTranscript.metadata.model_info.detection_model}</p>
                )}
                {viewingTranscript.metadata?.model_info?.layout_model && (
                  <p><span className="font-semibold text-gray-700">Layout:</span> {viewingTranscript.metadata.model_info.layout_model}</p>
                )}
                {(viewingTranscript.metadata?.model_info?.ocr_provider || viewingTranscript.metadata?.model_info?.ocr_model) && (
                  <p><span className="font-semibold text-gray-700">OCR:</span> {viewingTranscript.metadata.model_info.ocr_provider} / {viewingTranscript.metadata.model_info.ocr_model}</p>
                )}
                {viewingTranscript.metadata?.model_info?.llm_postprocess && (
                  <p><span className="font-semibold text-gray-700">LLM:</span> {viewingTranscript.metadata.model_info.llm_postprocess.used ? `${viewingTranscript.metadata.model_info.llm_postprocess.provider} / ${viewingTranscript.metadata.model_info.llm_postprocess.model}` : 'none'}</p>
                )}
                <p><span className="font-semibold text-gray-700">ID:</span> {viewingTranscript.metadata?.id}</p>
              </div>

              {(viewingTranscript.pages || []).map((page) => (
                <article key={page.name} className="border border-blue-100 rounded-xl overflow-hidden">
                  <div className="px-3 py-2 bg-blue-50 text-xs font-semibold text-blue-700">{page.name}</div>
                  {(() => {
                    const previewHeight = computeDynamicPreviewHeight(transcriptImageRatios[page.name]);
                    return (
                  <div className="grid md:grid-cols-2 items-stretch gap-0">
                    <div className="border-b md:border-b-0 md:border-r border-blue-100 bg-slate-50" style={{ height: `${previewHeight}px` }}>
                      {page.image_data ? (
                        <img
                          src={page.image_data}
                          alt={page.name}
                          className="w-full h-full object-contain"
                          onLoad={(event) => handleTranscriptImageLoad(page.name, event)}
                        />
                      ) : (
                        <div className="h-full flex flex-col items-center justify-center text-slate-500">
                          <ImageIcon className="w-5 h-5 mb-1" />
                          <p className="text-xs font-medium">No saved page image</p>
                        </div>
                      )}
                    </div>
                    <div className="p-3 text-sm text-gray-700 bg-white flex flex-col" style={{ height: `${previewHeight}px` }}>
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Transcript</p>
                      <pre className="whitespace-pre-wrap text-sm leading-relaxed flex-1 overflow-y-auto">{page.content}</pre>
                    </div>
                  </div>
                    );
                  })()}
                </article>
              ))}
              {(!viewingTranscript.pages || viewingTranscript.pages.length === 0) && (
                <p className="text-sm text-gray-500">No transcript pages found.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {viewingDataset && (
        <div className="fixed inset-0 z-50 bg-gray-900/45 backdrop-blur-sm p-4 flex items-center justify-center">
          <div className="w-full max-w-5xl max-h-[86vh] bg-white rounded-2xl shadow-2xl border border-gray-100 flex flex-col overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
              <div>
                <p className="font-semibold text-gray-800">{viewingDataset.metadata?.book_name || viewingDataset.metadata?.id}</p>
                <p className="text-xs text-gray-500">{formatDate(viewingDataset.metadata?.created_at)}</p>
                <TagChips tags={viewingDataset.metadata?.tags} />
              </div>
              <button
                onClick={() => setViewingDataset(null)}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-4 space-y-4 overflow-y-auto">
              <div className="grid md:grid-cols-3 gap-2 text-xs text-gray-600 bg-gray-50 rounded-xl p-3 border border-gray-100">
                <p><span className="font-semibold text-gray-700">Mode:</span> {viewingDataset.metadata?.dataset_type || viewingDataset.metadata?.mode}</p>
                <p><span className="font-semibold text-gray-700">Samples:</span> {viewingDataset.metadata?.num_samples || viewingDataset.metadata?.num_files || 0}</p>
                <p><span className="font-semibold text-gray-700">Format:</span> {viewingDataset.metadata?.format || 'unknown'}</p>
                {viewingDataset.metadata?.model_info?.preprocessing?.length > 0 && (
                  <p className="md:col-span-2"><span className="font-semibold text-gray-700">Preprocessing:</span> {viewingDataset.metadata.model_info.preprocessing.join(', ')}</p>
                )}
                {viewingDataset.metadata?.model_info?.detection_model && (
                  <p><span className="font-semibold text-gray-700">Detection:</span> {viewingDataset.metadata.model_info.detection_model}</p>
                )}
                {viewingDataset.metadata?.model_info?.layout_model && (
                  <p><span className="font-semibold text-gray-700">Layout:</span> {viewingDataset.metadata.model_info.layout_model}</p>
                )}
                {(viewingDataset.metadata?.model_info?.ocr_provider || viewingDataset.metadata?.model_info?.ocr_model) && (
                  <p><span className="font-semibold text-gray-700">OCR:</span> {viewingDataset.metadata.model_info.ocr_provider} / {viewingDataset.metadata.model_info.ocr_model}</p>
                )}
                {viewingDataset.metadata?.model_info?.llm_postprocess && (
                  <p><span className="font-semibold text-gray-700">LLM:</span> {viewingDataset.metadata.model_info.llm_postprocess.used ? `${viewingDataset.metadata.model_info.llm_postprocess.provider} / ${viewingDataset.metadata.model_info.llm_postprocess.model}` : 'none'}</p>
                )}
                <p><span className="font-semibold text-gray-700">Source:</span> {viewingDataset.metadata?.source || 'n/a'}</p>
                <p><span className="font-semibold text-gray-700">ID:</span> {viewingDataset.metadata?.id}</p>
              </div>

              <div className="space-y-3">
                {(viewingDataset.samples || []).length === 0 && (
                  <p className="text-sm text-gray-500">No dataset preview samples found.</p>
                )}

                {(viewingDataset.samples || []).map((sample) => (
                  <article key={sample.name} className="border border-emerald-100 rounded-xl overflow-hidden">
                    <div className="px-3 py-2 bg-emerald-50 text-xs font-semibold text-emerald-700">{sample.name}</div>
                    {(() => {
                      const previewHeight = computeDynamicPreviewHeight(datasetImageRatios[sample.name]);
                      return (
                    <div className="grid md:grid-cols-2 items-stretch">
                      <div className="border-b md:border-b-0 md:border-r border-emerald-100 bg-slate-50" style={{ height: `${previewHeight}px` }}>
                        {sample.image_data ? (
                          <img
                            src={sample.image_data}
                            alt={sample.name}
                            className="w-full h-full object-contain"
                            onLoad={(event) => handleDatasetImageLoad(sample.name, event)}
                          />
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center text-slate-500">
                            <ImageIcon className="w-5 h-5 mb-1" />
                            <p className="text-xs font-medium">No sample image</p>
                          </div>
                        )}
                      </div>
                      <div className="p-3 text-sm text-gray-700 bg-white flex flex-col" style={{ height: `${previewHeight}px` }}>
                        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">Label / Annotation</p>
                        <pre className="whitespace-pre-wrap text-sm leading-relaxed flex-1 overflow-y-auto">
                          {sample.text || sample.annotation || 'No annotation text'}
                        </pre>
                      </div>
                    </div>
                      );
                    })()}
                  </article>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
