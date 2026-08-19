import React from 'react';
import { Check, Upload, Layers, Wand2, ScanText, Sparkles } from 'lucide-react';

const steps = [
  { id: 1, name: 'Upload', description: 'Upload PDF or images', icon: Upload },
  { id: 2, name: 'Select Pages', description: 'Choose pages to process', icon: Layers },
  { id: 3, name: 'Preprocess', description: 'Apply image transformations', icon: Wand2 },
  { id: 4, name: 'Text Detection', description: 'Choose detection method', icon: ScanText },
  { id: 5, name: 'OCR', description: 'Extract text with Gemini', icon: Sparkles },
];

export default function Stepper({ currentStep, onStepClick }) {
  const progressPercent = ((currentStep - 1) / (steps.length - 1)) * 100;

  return (
    <div className="w-full bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/50 p-6 mb-6">
      <nav aria-label="Progress">
        {/* Progress bar background */}
        <div className="relative">
          {/* Background line */}
          <div className="absolute top-5 left-[10%] right-[10%] h-1 bg-gradient-to-r from-blue-100 via-indigo-100 to-blue-100 rounded-full" />

          {/* Filled progress line */}
          <div
            className="absolute top-5 left-[10%] h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-blue-600 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progressPercent * 0.8}%` }}
          />

          <ol className="relative flex items-center justify-between">
            {steps.map((step, stepIdx) => {
              const isCompleted = currentStep > step.id;
              const isCurrent = currentStep === step.id;
              const isDisabled = step.disabled;
              const isClickable = !isDisabled && step.id <= currentStep;
              const StepIcon = step.icon;

              return (
                <li key={step.id} className="flex-1 flex flex-col items-center">
                  <button
                    onClick={() => isClickable && onStepClick(step.id)}
                    disabled={!isClickable}
                    className={`relative flex flex-col items-center group transition-all duration-200 ${isClickable ? 'cursor-pointer' : 'cursor-not-allowed'
                      }`}
                  >
                    {/* Step circle with icon */}
                    <span
                      className={`relative w-10 h-10 flex items-center justify-center rounded-full border-2 transition-all duration-300 ${isCompleted
                          ? 'bg-gradient-to-br from-blue-500 to-indigo-600 border-blue-500 text-white shadow-lg shadow-blue-500/30'
                          : isCurrent
                            ? 'bg-white border-blue-500 text-blue-600 ring-4 ring-blue-100 shadow-md animate-pulse-soft'
                            : isDisabled
                              ? 'bg-gray-100 border-gray-200 text-gray-400'
                              : 'bg-white border-gray-200 text-gray-400 group-hover:border-blue-300 group-hover:text-blue-500 group-hover:shadow-md'
                        }`}
                    >
                      {isCompleted ? (
                        <Check className="w-5 h-5" strokeWidth={2.5} />
                      ) : (
                        <StepIcon className={`w-5 h-5 transition-transform duration-200 ${isCurrent ? 'scale-110' : 'group-hover:scale-110'
                          }`} />
                      )}

                      {/* Glow effect on current step */}
                      {isCurrent && (
                        <span className="absolute inset-0 rounded-full bg-blue-400/20 animate-ping" />
                      )}
                    </span>

                    {/* Step name */}
                    <span
                      className={`mt-3 text-sm font-semibold transition-colors duration-200 ${isCurrent
                          ? 'text-blue-600'
                          : isCompleted
                            ? 'text-blue-700'
                            : isDisabled
                              ? 'text-gray-400'
                              : 'text-gray-500 group-hover:text-blue-500'
                        }`}
                    >
                      {step.name}
                    </span>

                    {/* Step description (hidden on small screens) */}
                    <span
                      className={`hidden md:block text-xs mt-1 transition-colors duration-200 max-w-24 text-center ${isCurrent
                          ? 'text-blue-500'
                          : isDisabled
                            ? 'text-gray-300'
                            : 'text-gray-400'
                        }`}
                    >
                      {isDisabled ? 'Coming soon' : step.description}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </div>
      </nav>
    </div>
  );
}
