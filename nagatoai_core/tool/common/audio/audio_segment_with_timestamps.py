# Third Party
from pydantic import BaseModel
from pydub import AudioSegment


class AudioSegmentWithOffsets(BaseModel):
    """
    AudioSegmentWithOffsets is a class that contains an AudioSegment and the start and end offsets of the segment.
    """

    audio: AudioSegment
    from_second_offset: float
    to_second_offset: float

    class Config:
        arbitrary_types_allowed = True  # Required to allow AudioSegment as a field type
