import os
from typing import Type
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
import assemblyai as aai

from nagatoai_core.tool.abstract_tool import AbstractTool


class AssemblyAITranscriptionConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The AssemblyAI API key to use for authentication.",
        alias="ASSEMBLYAI_API_KEY",
        exclude_from_schema=True,
    )

    media_full_path: str = Field(
        ...,
        description="The full path to the media file (video or audio) to transcribe.",
    )

    speaker_labels: bool = Field(
        True,
        description="Whether to turn on speaker diarization. Default is True.",
    )

    speech_model: str = Field(
        "aai.SpeechModel.best",
        description="The speech model to use for transcription. Default is 'aai.SpeechModel.best'.",
    )


class AssemblyAITranscriptionTool(AbstractTool):
    """
    AssemblyAITranscriptionTool is a tool that transcribes audio/video files using AssemblyAI.
    """

    name: str = "assemblyai_transcription"
    description: str = (
        """Transcribes an audio/video file using AssemblyAI.
        Returns the transcription text and additional metadata.
        """
    )
    args_schema: Type[BaseModel] = AssemblyAITranscriptionConfig

    def _run(self, config: AssemblyAITranscriptionConfig) -> dict:
        """
        Transcribes an audio/video file using AssemblyAI.
        :param config: The configuration for the AssemblyAITranscriptionTool.
        :return: A dictionary containing the transcription text and additional metadata.
        """
        try:
            if not os.path.exists(config.media_full_path):
                return {"error": "File not found", "file_path": config.media_full_path}

            # Set up the AssemblyAI client
            aai.settings.api_key = config.api_key

            # Create a transcriber
            transcriber = aai.Transcriber()

            # Set up the configuration
            transcript_config = aai.TranscriptionConfig(
                speaker_labels=config.speaker_labels,
                speech_model=eval(
                    config.speech_model
                ),  # Convert string to actual enum value
            )

            # Start the transcription
            transcript = transcriber.transcribe(
                config.media_full_path, config=transcript_config
            )

            # Prepare the result
            result = {
                "text": transcript.text,
                "audio_duration": transcript.audio_duration,
                "confidence": transcript.confidence,
                "words": [
                    {
                        "text": w.text,
                        "start_seconds": w.start / 1000.0,
                        "end_seconds": w.end / 1000.0,
                        "confidence": w.confidence,
                    }
                    for w in transcript.words
                ],
                "file_name": os.path.basename(config.media_full_path),
                "speech_model": config.speech_model,
                "speaker_labels": config.speaker_labels,
            }

            # Add speaker labels if enabled
            if config.speaker_labels:
                result["utterances"] = [
                    {
                        "speaker": u.speaker,
                        "text": u.text,
                        "start_seconds": u.start / 1000.0,
                        "end_seconds": u.end / 1000.0,
                    }
                    for u in transcript.utterances
                ]

            sentences = transcript.get_sentences()
            sentences_text = []
            for sentence in sentences:
                sentences_text.append(
                    {
                        "text": sentence.text,
                        "start_seconds": sentence.start / 1000.0,
                        "end_seconds": sentence.end / 1000.0,
                    }
                )
            result["sentences"] = sentences_text

            return result

        except Exception as e:
            raise RuntimeError(f"Error transcribing audio/video file: {str(e)}")
