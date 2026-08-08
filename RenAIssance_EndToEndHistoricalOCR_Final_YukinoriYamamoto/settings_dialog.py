"""
Settings Dialog for OCR Enhancement
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk


class SettingsDialog:
    def __init__(self, parent):
        self.parent = parent
        self.result = None

        # Load existing settings
        self.settings = self.load_settings()

        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("OCR Enhancement Settings")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.geometry(
            "+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self.create_widgets()

    def load_settings(self):
        """Load settings from file"""
        settings_file = "ocr_settings.json"
        default_settings = {
            "openai_api_key": "",
            "enhancement_iterations": 3,
            "low_confidence_tokens": 10,
            "model_name": "gpt-4o-mini",
        }

        try:
            if os.path.exists(settings_file):
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    # Merge with defaults to handle new settings
                    for key, value in default_settings.items():
                        if key not in settings:
                            settings[key] = value
                    return settings
        except Exception as e:
            print(f"Error loading settings: {e}")

        return default_settings

    def save_settings(self):
        """Save settings to file"""
        settings_file = "ocr_settings.json"
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(
            main_frame, text="OCR Enhancement Settings", style="Heading.TLabel"
        )
        title_label.pack(pady=(0, 20))

        # OpenAI API Key
        api_frame = ttk.LabelFrame(main_frame, text="OpenAI Configuration")
        api_frame.pack(fill=tk.X, pady=(0, 15))

        api_key_frame = ttk.Frame(api_frame)
        api_key_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(api_key_frame, text="API Key:").pack(anchor=tk.W)
        self.api_key_var = tk.StringVar(value=self.settings.get("openai_api_key", ""))
        self.api_key_entry = ttk.Entry(
            api_key_frame, textvariable=self.api_key_var, show="*", width=50
        )
        self.api_key_entry.pack(fill=tk.X, pady=(5, 0))

        # Show/Hide API key button
        show_hide_frame = ttk.Frame(api_frame)
        show_hide_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.show_api_key = tk.BooleanVar()
        show_check = ttk.Checkbutton(
            show_hide_frame,
            text="Show API Key",
            variable=self.show_api_key,
            command=self.toggle_api_key_visibility,
        )
        show_check.pack(side=tk.LEFT)

        # Model selection
        model_frame = ttk.Frame(api_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(
            value=self.settings.get("model_name", "gpt-4o-mini")
        )
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            state="readonly",
            width=20,
        )
        model_combo.pack(side=tk.RIGHT)

        # Enhancement Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Enhancement Parameters")
        params_frame.pack(fill=tk.X, pady=(0, 15))

        # Iterations
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(iter_frame, text="Enhancement Iterations:").pack(side=tk.LEFT)
        self.iterations_var = tk.IntVar(
            value=self.settings.get("enhancement_iterations", 3)
        )
        iter_spinbox = ttk.Spinbox(
            iter_frame, from_=1, to=10, textvariable=self.iterations_var, width=5
        )
        iter_spinbox.pack(side=tk.RIGHT)

        # Low confidence tokens
        conf_frame = ttk.Frame(params_frame)
        conf_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(conf_frame, text="Low Confidence Tokens to Highlight:").pack(
            side=tk.LEFT
        )
        self.confidence_var = tk.IntVar(
            value=self.settings.get("low_confidence_tokens", 10)
        )
        conf_spinbox = ttk.Spinbox(
            conf_frame, from_=1, to=100, textvariable=self.confidence_var, width=5
        )
        conf_spinbox.pack(side=tk.RIGHT)

        # Help text
        help_frame = ttk.LabelFrame(main_frame, text="Instructions")
        help_frame.pack(fill=tk.X, pady=(0, 15))

        help_text = """1. Enter your OpenAI API key (required for enhancement)
2. Choose the number of enhancement iterations (more = better quality, slower)
3. Set how many low-confidence tokens to highlight in red
4. Enhancement uses ROVER algorithm to combine multiple AI corrections"""

        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT)
        help_label.pack(padx=10, pady=10, anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(button_frame, text="Save", command=self.save).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Test API Key", command=self.test_api_key).pack(
            side=tk.LEFT
        )

    def toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_api_key.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")

    def test_api_key(self):
        """Test the OpenAI API key"""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first.")
            return

        try:
            # Test the API key with a simple request
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            # Simple test request
            response = client.chat.completions.create(
                model=self.model_var.get(),
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )

            messagebox.showinfo("Success", "API key is valid!")

        except Exception as e:
            messagebox.showerror("Error", f"API key test failed: {str(e)}")

    def save(self):
        """Save settings and close dialog"""
        # Validate API key
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an OpenAI API key.")
            return

        # Update settings
        self.settings.update(
            {
                "openai_api_key": api_key,
                "enhancement_iterations": self.iterations_var.get(),
                "low_confidence_tokens": self.confidence_var.get(),
                "model_name": self.model_var.get(),
            }
        )

        # Save to file
        self.save_settings()

        self.result = self.settings
        self.dialog.destroy()

    def cancel(self):
        """Cancel and close dialog"""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Show dialog and return result"""
        self.dialog.wait_window()
        return self.result


def show_settings_dialog(parent):
    """Convenience function to show settings dialog"""
    dialog = SettingsDialog(parent)
    return dialog.show()
