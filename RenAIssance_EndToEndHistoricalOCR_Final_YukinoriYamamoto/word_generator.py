"""
Word Document Generator
Handles creation of Word documents with confidence-based highlighting
"""

import os
import re
from typing import Any, Dict, List, Tuple

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import RGBColor


class ConfidenceWordGenerator:
    """
    Generates Word documents with confidence-based text highlighting
    """

    def __init__(self):
        pass

    def calculate_low_confidence_tokens(
        self, enhanced_results: List[Dict[str, Any]], num_low_confidence: int
    ) -> List[Tuple[int, int]]:
        """
        Calculate which tokens have the lowest confidence across all lines

        Args:
            enhanced_results: List of enhanced OCR results
            num_low_confidence: Number of tokens to identify as low confidence

        Returns:
            List of (line_index, token_index) tuples for low confidence tokens
        """
        all_confidences = []

        for line_idx, line_data in enumerate(enhanced_results):
            enhanced_text = line_data.get("enhanced_text", "") or line_data.get(
                "consensus_text", ""
            )
            confidence_scores = line_data.get("confidence_scores", [])

            if not enhanced_text or not confidence_scores:
                continue

            tokens = self.tokenize_text(enhanced_text)
            for token_idx, confidence in enumerate(confidence_scores):
                if token_idx < len(tokens):
                    all_confidences.append((confidence, line_idx, token_idx))

        # Sort by confidence (lowest first)
        all_confidences.sort(key=lambda x: x[0])

        # Get the positions of the lowest confidence tokens
        low_confidence_positions = []
        for i in range(min(num_low_confidence, len(all_confidences))):
            _, line_idx, token_idx = all_confidences[i]
            low_confidence_positions.append((line_idx, token_idx))

        return low_confidence_positions

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words and punctuation"""
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return [token for token in tokens if token.strip()]

    def sort_results_by_position(
        self, enhanced_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort enhanced results by vertical position (y_position)

        Args:
            enhanced_results: List of enhanced OCR results

        Returns:
            Sorted list of enhanced OCR results
        """
        # Sort by y_position (top to bottom)
        sorted_results = sorted(enhanced_results, key=lambda x: x.get("y_position", 0))
        return sorted_results

    def create_word_document(
        self,
        enhanced_results: List[Dict[str, Any]],
        num_low_confidence: int,
        output_path: str,
        document_title: str = "OCR Transcription with Confidence Analysis",
    ) -> bool:
        """
        Create a Word document with confidence-based highlighting

        Args:
            enhanced_results: List of enhanced OCR results
            num_low_confidence: Number of lowest confidence tokens to highlight
            output_path: Path to save the Word document
            document_title: Title for the document

        Returns:
            True if successful, False otherwise
        """
        try:
            # Sort results by vertical position first
            sorted_results = self.sort_results_by_position(enhanced_results)

            # Calculate low confidence tokens
            low_confidence_positions = self.calculate_low_confidence_tokens(
                sorted_results, num_low_confidence
            )

            # Create new document
            doc = Document()

            # Add title
            doc.add_heading(document_title, 0)

            # Add transcription section heading
            doc.add_heading("転写結果 (Transcription Results)", 1)

            # Add metadata
            doc.add_paragraph(
                f"処理されたライン数 (Lines processed): {len(sorted_results)}"
            )
            doc.add_paragraph(
                f"低信頼度トークン数 (Low confidence tokens): {num_low_confidence}"
            )

            # Add analysis section heading
            doc.add_heading("詳細分析 (Detailed Analysis)", 1)

            # Add statistics
            total_tokens = sum(
                len(
                    self.tokenize_text(
                        result.get("enhanced_text", "")
                        or result.get("consensus_text", "")
                    )
                )
                for result in sorted_results
            )
            doc.add_paragraph(f"総トークン数 (Total tokens): {total_tokens}")

            if sorted_results:
                avg_confidence = sum(
                    sum(result.get("confidence_scores", []))
                    / max(len(result.get("confidence_scores", [])), 1)
                    for result in sorted_results
                ) / len(sorted_results)
                doc.add_paragraph(
                    f"平均信頼度 (Average confidence): {avg_confidence:.3f}"
                )

            # Add per-line details
            for line_idx, line_data in enumerate(sorted_results):
                enhanced_text = line_data.get("enhanced_text", "") or line_data.get(
                    "consensus_text", ""
                )
                confidence_scores = line_data.get("confidence_scores", [])
                y_position = line_data.get("y_position", 0)

                if not enhanced_text:
                    continue

                # Add line number and position
                line_paragraph = doc.add_paragraph()
                line_paragraph.add_run(
                    f"Line {line_idx + 1} (Y={y_position}): "
                ).bold = True

                # Tokenize and add text with highlighting
                tokens = self.tokenize_text(enhanced_text)
                for token_idx, token in enumerate(tokens):
                    run = line_paragraph.add_run(token)

                    # Check if this token should be highlighted
                    if (line_idx, token_idx) in low_confidence_positions:
                        run.font.highlight_color = WD_COLOR_INDEX.YELLOW

                    # Add space after token (except for punctuation)
                    if token_idx < len(tokens) - 1 and not re.match(r"[^\w\s]", token):
                        line_paragraph.add_run(" ")

                # Add confidence info
                if confidence_scores:
                    avg_line_confidence = sum(confidence_scores) / len(
                        confidence_scores
                    )
                    conf_paragraph = doc.add_paragraph()
                    conf_run = conf_paragraph.add_run(
                        f"  → 平均信頼度 (Avg confidence): {avg_line_confidence:.3f}"
                    )
                    conf_run.font.color.rgb = RGBColor(128, 128, 128)  # Gray

            # Save document
            doc.save(output_path)
            print(f"Word document saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error creating Word document: {e}")
            return False

    def create_transcription_only_document(
        self,
        enhanced_results: List[Dict[str, Any]],
        output_path: str,
        document_title: str = "OCR転写結果 (OCR Transcription Results)",
        append_mode: bool = False,
        page_info: str = None,
    ) -> bool:
        """
        Create a Word document with just the transcription text, sorted by vertical position

        Args:
            enhanced_results: List of enhanced OCR results
            output_path: Path to save the Word document
            document_title: Title for the document
            append_mode: Whether to append to existing document or create new one
            page_info: Page information for multi-page documents

        Returns:
            True if successful, False otherwise
        """
        try:
            print(
                f"DEBUG: Creating transcription document with {len(enhanced_results)} results"
            )
            # Sort results by vertical position first
            sorted_results = self.sort_results_by_position(enhanced_results)

            # Load existing document or create new one
            if append_mode and os.path.exists(output_path):
                doc = Document(output_path)

                # Add page separator if this is a new page
                if page_info:
                    doc.add_page_break()
                    doc.add_heading(f"{document_title} - {page_info}", 1)
                else:
                    doc.add_paragraph("")  # Add some spacing
                    doc.add_heading("続き (Continued)", 2)
            else:
                # Create new document
                doc = Document()
                # Add title with page info if available
                full_title = (
                    f"{document_title} - {page_info}" if page_info else document_title
                )
                doc.add_heading(full_title, 0)

            # Add basic statistics for this page/section
            stats_text = f"処理されたライン数 (Lines processed): {len(sorted_results)}"
            if page_info:
                stats_text = f"{page_info} - {stats_text}"
            doc.add_paragraph(stats_text)
            doc.add_paragraph("")  # Empty line

            # Add transcription text line by line
            for line_idx, line_data in enumerate(sorted_results):
                enhanced_text = line_data.get("enhanced_text", "") or line_data.get(
                    "consensus_text", ""
                )

                if enhanced_text.strip():
                    paragraph = doc.add_paragraph()
                    # Add line number and y-position as a subtle reference
                    line_ref = paragraph.add_run(f"[{line_idx + 1}] ")
                    line_ref.font.color.rgb = RGBColor(128, 128, 128)  # Gray
                    line_ref.font.size = (
                        line_ref.font.size * 0.8 if line_ref.font.size else None
                    )

                    # Add the main text
                    paragraph.add_run(enhanced_text)

            # Save document
            doc.save(output_path)
            print(f"Transcription document saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error creating transcription document: {e}")
            return False

    def create_detailed_analysis_document(
        self,
        enhanced_results: List[Dict[str, Any]],
        num_low_confidence: int,
        output_path: str,
        document_title: str = "OCR詳細分析 (OCR Detailed Analysis)",
        page_info: str = None,
    ) -> bool:
        """
        Create a Word document with detailed analysis and confidence highlighting

        Args:
            enhanced_results: List of enhanced OCR results
            num_low_confidence: Number of lowest confidence tokens to highlight
            output_path: Path to save the Word document
            document_title: Title for the document
            page_info: Page information for multi-page documents

        Returns:
            True if successful, False otherwise
        """
        try:
            # Sort results by vertical position first
            sorted_results = self.sort_results_by_position(enhanced_results)

            # Calculate low confidence tokens
            low_confidence_positions = self.calculate_low_confidence_tokens(
                sorted_results, num_low_confidence
            )

            # Create new document
            doc = Document()

            # Add title with page info if available
            full_title = (
                f"{document_title} - {page_info}" if page_info else document_title
            )
            doc.add_heading(full_title, 0)

            # Add summary statistics
            doc.add_heading("処理統計 (Processing Statistics)", 1)
            total_tokens = sum(
                len(
                    self.tokenize_text(
                        result.get("enhanced_text", "")
                        or result.get("consensus_text", "")
                    )
                )
                for result in sorted_results
            )
            doc.add_paragraph(
                f"処理されたライン数 (Lines processed): {len(sorted_results)}"
            )
            doc.add_paragraph(f"総トークン数 (Total tokens): {total_tokens}")
            doc.add_paragraph(
                f"低信頼度トークン数 (Low confidence tokens): {num_low_confidence}"
            )

            if sorted_results:
                avg_confidence = sum(
                    sum(result.get("confidence_scores", []))
                    / max(len(result.get("confidence_scores", [])), 1)
                    for result in sorted_results
                ) / len(sorted_results)
                doc.add_paragraph(
                    f"平均信頼度 (Average confidence): {avg_confidence:.3f}"
                )

            # Add legend
            doc.add_heading("凡例 (Legend)", 1)
            legend_p = doc.add_paragraph("黄色ハイライト (Yellow highlight): ")
            highlighted_run = legend_p.add_run(
                "低信頼度トークン (Low confidence tokens)"
            )
            highlighted_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

            # Add detailed line-by-line analysis
            doc.add_heading("行ごとの詳細分析 (Line-by-line Analysis)", 1)

            for line_idx, line_data in enumerate(sorted_results):
                enhanced_text = line_data.get("enhanced_text", "") or line_data.get(
                    "consensus_text", ""
                )
                confidence_scores = line_data.get("confidence_scores", [])
                y_position = line_data.get("y_position", 0)
                original_text = line_data.get("original_text", "")

                if not enhanced_text:
                    continue

                # Add line header
                doc.add_heading(f"Line {line_idx + 1} (Y位置: {y_position})", 2)

                # Add original vs enhanced comparison if available
                if original_text and original_text != enhanced_text:
                    doc.add_paragraph().add_run(
                        "元のOCR結果 (Original OCR): "
                    ).bold = True
                    doc.add_paragraph(original_text)
                    doc.add_paragraph().add_run(
                        "拡張結果 (Enhanced result): "
                    ).bold = True
                else:
                    doc.add_paragraph().add_run(
                        "転写結果 (Transcription): "
                    ).bold = True

                # Add enhanced text with highlighting
                text_paragraph = doc.add_paragraph()
                tokens = self.tokenize_text(enhanced_text)
                for token_idx, token in enumerate(tokens):
                    run = text_paragraph.add_run(token)

                    # Highlight low confidence tokens
                    if (line_idx, token_idx) in low_confidence_positions:
                        run.font.highlight_color = WD_COLOR_INDEX.YELLOW

                    # Add space after token (except for punctuation)
                    if token_idx < len(tokens) - 1 and not re.match(r"[^\w\s]", token):
                        text_paragraph.add_run(" ")

                # Add confidence statistics for this line
                if confidence_scores:
                    avg_line_confidence = sum(confidence_scores) / len(
                        confidence_scores
                    )
                    min_confidence = min(confidence_scores)
                    max_confidence = max(confidence_scores)

                    stats_p = doc.add_paragraph()
                    stats_run = stats_p.add_run(
                        f"信頼度統計 (Confidence stats): 平均={avg_line_confidence:.3f}, "
                        f"最小={min_confidence:.3f}, 最大={max_confidence:.3f}"
                    )
                    stats_run.font.color.rgb = RGBColor(100, 100, 100)  # Dark gray

                # Add spacing between lines
                doc.add_paragraph("")

            # Save document
            doc.save(output_path)
            print(f"Detailed analysis document saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error creating detailed analysis document: {e}")
            return False
