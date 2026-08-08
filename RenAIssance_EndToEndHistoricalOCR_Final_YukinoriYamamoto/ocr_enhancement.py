"""
OCR Enhancement Module
Handles OpenAI API integration, ROVER consensus, and confidence calculation
"""

import asyncio
import base64
import io
import random
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from openai import AsyncOpenAI
from PIL import Image


class ROVERConsensus:
    """
    ROVER (Recognizer Output Voting Error Reduction) implementation
    for OCR consensus and confidence calculation
    """

    def __init__(self):
        self.word_separator = " "

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Basic tokenization - split by whitespace and punctuation
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return [token for token in tokens if token.strip()]

    def align_sequences(self, sequences: List[List[str]]) -> List[List[str]]:
        """
        Simple alignment of token sequences
        More sophisticated alignment could use edit distance
        """
        if not sequences:
            return []

        # Find the longest sequence as reference
        reference = max(sequences, key=len)
        aligned = [reference]

        for seq in sequences:
            if seq != reference:
                aligned_seq = self._align_to_reference(reference, seq)
                aligned.append(aligned_seq)

        return aligned

    def _align_to_reference(
        self, reference: List[str], sequence: List[str]
    ) -> List[str]:
        """Align a sequence to a reference sequence"""
        if len(sequence) == len(reference):
            return sequence

        # Simple alignment: pad shorter sequences with empty strings
        if len(sequence) < len(reference):
            sequence = sequence + [""] * (len(reference) - len(sequence))
        else:
            # Truncate longer sequences
            sequence = sequence[: len(reference)]

        return sequence

    def calculate_consensus_and_confidence(
        self, transcriptions: List[str]
    ) -> Tuple[str, List[float]]:
        """
        Calculate consensus transcription and per-token confidence

        Returns:
            Tuple of (consensus_text, confidence_scores)
        """
        if not transcriptions:
            return "", []

        if len(transcriptions) == 1:
            tokens = self.tokenize(transcriptions[0])
            return transcriptions[0], [1.0] * len(tokens)

        # Tokenize all transcriptions
        tokenized = [self.tokenize(trans) for trans in transcriptions]

        # Align sequences
        aligned = self.align_sequences(tokenized)

        # Calculate consensus for each position
        consensus_tokens = []
        confidence_scores = []

        max_length = max(len(seq) for seq in aligned) if aligned else 0

        for pos in range(max_length):
            # Get all tokens at this position
            position_tokens = []
            for seq in aligned:
                if pos < len(seq) and seq[pos]:
                    position_tokens.append(seq[pos])

            if not position_tokens:
                continue

            # Count occurrences
            token_counts = Counter(position_tokens)

            # Most common token is the consensus
            consensus_token = token_counts.most_common(1)[0][0]
            consensus_tokens.append(consensus_token)

            # Confidence is the ratio of consensus votes to total votes
            consensus_count = token_counts[consensus_token]
            total_votes = len(position_tokens)
            confidence = consensus_count / total_votes
            confidence_scores.append(confidence)

        consensus_text = self.word_separator.join(consensus_tokens)
        return consensus_text, confidence_scores


class OpenAIEnhancer:
    """
    OpenAI API integration for OCR enhancement
    """

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.prompt_template = (
            "You are an excellent proofreader specializing in historical Spanish documents with handwritten text. "
            "This image contains a single line of handwritten Spanish. "
            'The current transcription is "{transcription}". '
            "Carefully examine the image and provide ONLY the corrected transcription of the text line. "
            "Do not include any explanations or additional text."
        )
        self.total_tokens_used = 0  # Track token usage for rate limiting

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count for rate limiting"""
        # Rough approximation: 1 token ≈ 4 characters for English/Spanish
        # Add buffer for system message and image tokens (images are expensive)
        return len(text) // 4 + 500  # Conservative estimate including image tokens

    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return image_base64

    async def enhance_single_transcription(
        self, image: np.ndarray, initial_transcription: str, max_retries: int = 3
    ) -> str:
        """
        Enhance a single transcription using OpenAI Vision API with retry logic
        """
        for attempt in range(max_retries):
            try:
                # Convert image to base64
                image_base64 = self.image_to_base64(image)

                # Prepare the prompt
                prompt = self.prompt_template.format(
                    transcription=initial_transcription
                )

                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Using GPT-4 Vision
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                    temperature=0.1,  # Low temperature for consistent results
                )

                enhanced_text = response.choices[0].message.content.strip()
                return enhanced_text

            except Exception as e:
                error_str = str(e)
                print(f"DEBUG: OpenAI API error on attempt {attempt + 1}: {error_str}")
                is_rate_limit = "rate limit" in error_str.lower() or "429" in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    # More aggressive exponential backoff for rate limiting
                    base_delay = (3**attempt) + random.uniform(
                        2, 5
                    )  # Start with longer delays

                    # Extract wait time from error message if available
                    if "try again in" in error_str:
                        try:
                            import re

                            match = re.search(r"try again in (\d+)ms", error_str)
                            if match:
                                wait_ms = int(match.group(1))
                                suggested_delay = wait_ms / 1000.0  # Convert to seconds
                                # Add buffer time
                                base_delay = max(base_delay, suggested_delay + 2.0)
                        except Exception:
                            pass

                    print(
                        f"Rate limit hit, retrying in {base_delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(base_delay)
                elif attempt < max_retries - 1:
                    # Other errors - longer delay to be safe
                    delay = 2 + random.uniform(0, 2)
                    print(
                        f"API error, retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    print(
                        f"Error enhancing transcription after {max_retries} attempts: {e}"
                    )
                    return initial_transcription  # Return original if enhancement fails

    async def enhance_multiple_times(
        self, image: np.ndarray, initial_transcription: str, num_iterations: int
    ) -> List[str]:
        """
        Enhance transcription multiple times to get variations
        """
        enhanced_transcriptions = []

        # Process requests sequentially with longer delays to avoid rate limiting
        for i in range(num_iterations):
            try:
                result = await self.enhance_single_transcription(
                    image, initial_transcription
                )
                enhanced_transcriptions.append(result)

                # Add longer delay between requests to avoid rate limiting
                if i < num_iterations - 1:  # Don't delay after the last request
                    # Increase delay significantly to avoid TPM limit
                    delay = random.uniform(3.0, 5.0)  # Random delay between 3-5 seconds
                    print(f"Waiting {delay:.1f} seconds before next enhancement...")
                    await asyncio.sleep(delay)

            except Exception as e:
                print(f"Enhancement request {i + 1} failed: {e}")
                # If there was an error, use the original transcription
                enhanced_transcriptions.append(initial_transcription)
                # Add a much longer delay after an error before retrying
                if i < num_iterations - 1:
                    print("Waiting 10 seconds after error before next attempt...")
                    await asyncio.sleep(10.0)

        return enhanced_transcriptions


class OCREnhancementPipeline:
    """
    Complete pipeline for OCR enhancement using OpenAI and ROVER
    """

    def __init__(self, openai_api_key: str):
        self.openai_enhancer = OpenAIEnhancer(openai_api_key)
        self.rover = ROVERConsensus()

    async def enhance_ocr_results(
        self,
        ocr_results: List[Dict[str, Any]],
        line_images: List[np.ndarray],
        num_iterations: int = 3,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Enhance OCR results for all text lines

        Args:
            ocr_results: List of OCR result dictionaries
            line_images: List of line images corresponding to OCR results
            num_iterations: Number of OpenAI API calls per line
            progress_callback: Optional callback function for progress updates

        Returns:
            Enhanced OCR results with consensus text and confidence scores
        """
        enhanced_results = []
        total_lines = len(ocr_results)

        for i, (ocr_result, line_image) in enumerate(zip(ocr_results, line_images)):
            current_line = i + 1
            print(f"Enhancing line {current_line}/{total_lines}...")

            if progress_callback:
                progress_callback(current_line, total_lines)

            initial_transcription = ocr_result["text"]
            print(
                f"DEBUG: Initial transcription for line {current_line}: '{initial_transcription}'"
            )
            print(f"DEBUG: Line image shape: {line_image.shape}")

            # Get multiple enhanced transcriptions
            enhanced_transcriptions = await self.openai_enhancer.enhance_multiple_times(
                line_image, initial_transcription, num_iterations
            )

            # Add original transcription to the mix
            all_transcriptions = [initial_transcription] + enhanced_transcriptions

            # Calculate consensus and confidence
            consensus_text, confidence_scores = (
                self.rover.calculate_consensus_and_confidence(all_transcriptions)
            )

            print(f"DEBUG: Consensus text for line {current_line}: '{consensus_text}'")
            print(f"DEBUG: Confidence scores count: {len(confidence_scores)}")

            # Create enhanced result
            enhanced_result = ocr_result.copy()
            enhanced_result.update(
                {
                    "original_text": initial_transcription,
                    "enhanced_transcriptions": enhanced_transcriptions,
                    "consensus_text": consensus_text,
                    "enhanced_text": consensus_text,  # Add enhanced_text key for Word generator compatibility
                    "confidence_scores": confidence_scores,
                    "all_transcriptions": all_transcriptions,
                }
            )

            enhanced_results.append(enhanced_result)

            # Add delay between processing different lines to reduce rate limit pressure
            if i < total_lines - 1:  # Don't delay after the last line
                inter_line_delay = random.uniform(2.0, 4.0)  # 2-4 seconds between lines
                print(
                    f"Waiting {inter_line_delay:.1f} seconds before processing next line..."
                )
                await asyncio.sleep(inter_line_delay)

        return enhanced_results
