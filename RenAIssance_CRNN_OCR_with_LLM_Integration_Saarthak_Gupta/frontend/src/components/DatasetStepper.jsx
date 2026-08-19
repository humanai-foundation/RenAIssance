import React from 'react';
import {
  Check,
  Upload,
  GitMerge,
  Wand2,
  ScanText,
  Database,
} from 'lucide-react';

const recognitionSteps = [
  { id: 1, name: 'Upload', description: 'Book & transcript', icon: Upload },
  { id: 2, name: 'Match & Review', description: 'Verify page matches', icon: GitMerge },
  { id: 3, name: 'Preprocess', description: 'Apply image transforms', icon: Wand2 },
  { id: 4, name: 'Detect & Align', description: 'Detect lines & align text', icon: ScanText },
  { id: 5, name: 'Export Dataset', description: 'Download training data', icon: Database },
];

const detectionSteps = [
  { id: 1, name: 'Upload', description: 'Book images', icon: Upload },
  { id: 2, name: 'Select Pages', description: 'Pick pages to detect', icon: GitMerge },
  { id: 3, name: 'Preprocess', description: 'Apply image transforms', icon: Wand2 },
  { id: 4, name: 'Detect Lines', description: 'Detect text bounding boxes', icon: ScanText },
  { id: 5, name: 'Export Dataset', description: 'Download training data', icon: Database },
];

export default function DatasetStepper({ currentStep, onStepClick, compact = false, datasetMode = 'recognition' }) {
  const datasetSteps = datasetMode === 'detection' ? detectionSteps : recognitionSteps;
  const progressPercent = ((currentStep - 1) / (datasetSteps.length - 1)) * 100;

  const inner = (
    <nav aria-label="Progress">
      <div className="relative">
        {/* Background line — spans between the centres of the first and last step nodes */}
        <div className={`absolute ${compact ? 'top-3.5' : 'top-5'} left-[10%] right-[10%] h-0.5 bg-gradient-to-r from-emerald-100 via-teal-100 to-emerald-100 rounded-full`} />

        {/* Filled progress line */}
        <div
          className={`absolute ${compact ? 'top-3.5' : 'top-5'} left-[10%] h-0.5 bg-gradient-to-r from-emerald-500 via-teal-500 to-emerald-600 rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${progressPercent * 0.8}%` }}
        />

        <ol className="relative flex items-center justify-between">
          {datasetSteps.map((step) => {
            const isCompleted = currentStep > step.id;
            const isCurrent = currentStep === step.id;
            const isClickable = step.id <= currentStep;
            const StepIcon = step.icon;

            return (
              <li key={step.id} className="flex-1 flex flex-col items-center">
                <button
                  onClick={() => isClickable && onStepClick(step.id)}
                  disabled={!isClickable}
                  className={`relative flex flex-col items-center group transition-all duration-200 ${
                    isClickable ? 'cursor-pointer' : 'cursor-not-allowed'
                  }`}
                >
                  <span
                    className={`relative flex items-center justify-center rounded-full border-2 transition-all duration-300 ${
                      compact ? 'w-7 h-7' : 'w-10 h-10'
                    } ${
                      isCompleted
                        ? 'bg-gradient-to-br from-emerald-500 to-teal-600 border-emerald-500 text-white shadow-md shadow-emerald-500/30'
                        : isCurrent
                          ? 'bg-white border-emerald-500 text-emerald-600 ring-2 ring-emerald-100 shadow-md'
                          : 'bg-white border-gray-200 text-gray-400 group-hover:border-emerald-300 group-hover:text-emerald-500'
                    }`}
                  >
                    {isCompleted ? (
                      <Check className={compact ? 'w-3.5 h-3.5' : 'w-5 h-5'} strokeWidth={2.5} />
                    ) : (
                      <StepIcon
                        className={`${compact ? 'w-3.5 h-3.5' : 'w-5 h-5'} transition-transform duration-200 ${
                          isCurrent ? 'scale-110' : 'group-hover:scale-110'
                        }`}
                      />
                    )}

                    {isCurrent && (
                      <span className="absolute inset-0 rounded-full bg-emerald-400/20 animate-ping" />
                    )}
                  </span>

                  <span
                    className={`${compact ? 'mt-1 text-[10px]' : 'mt-3 text-xs'} font-semibold transition-colors duration-200 ${
                      isCurrent
                        ? 'text-emerald-600'
                        : isCompleted
                          ? 'text-emerald-700'
                          : 'text-gray-500 group-hover:text-emerald-500'
                    }`}
                  >
                    {step.name}
                  </span>

                  {!compact && (
                    <span
                      className={`hidden lg:block text-[10px] mt-1 transition-colors duration-200 max-w-20 text-center ${
                        isCurrent ? 'text-emerald-500' : 'text-gray-400'
                      }`}
                    >
                      {step.description}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </nav>
  );

  if (compact) {
    return <div className="w-full">{inner}</div>;
  }

  return (
    <div className="w-full bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 mb-6">
      {inner}
    </div>
  );
}
