import os
import pathlib
from typing import Dict, Optional, Union
from dataclasses import dataclass
import struct

@dataclass
class MemoryCell:
    """Represents a single addressable memory unit"""
    value: bytes = b'\x00'  # Default to zero byte
    dirty: bool = False

class WordAddressedMemory:
    """
    A word-addressed virtual memory system that maps to filesystem structure.
    Uses 16-bit word addressing (0x0000 to 0xFFFF) where:
    - First word (high byte): Directory address (0x00-0xFF)
    - Second word (low byte): File address (0x00-0xFF)
    - Third word: Offset within file (0x00-0xFF)
    """
    WORD_SIZE = 2  # 16 bits
    MAX_ADDRESS = 0xFFFF  # Maximum address in 16-bit space
    DIRECTORY_MASK = 0xFF00  # Mask for directory portion
    FILE_MASK = 0x00FF  # Mask for file portion
    
    def __init__(self, base_path: str = "/app/vmem"):
        self.base_path = pathlib.Path(base_path)
        self.memory_map: Dict[int, MemoryCell] = {}
        self.initialize_filesystem()
    
    def initialize_filesystem(self):
        """Create the directory structure based on word addressing"""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True)
            
        # Create directories for each possible high byte (0x00-0xFF)
        for dir_addr in range(0x100):
            dir_path = self.base_path / f"{dir_addr:02x}"
            if not dir_path.exists():
                dir_path.mkdir()
                
                # Create __init__.py in each directory
                init_file = dir_path / "__init__.py"
                init_content = f"""
# Auto-generated __init__.py for virtual memory directory 0x{dir_addr:02x}
from array import array

# Initialize memory array for this directory
memory_array = array('B', [0] * 256)  # 256 bytes (0x00-0xFF)

def get_byte(offset: int) -> int:
    return memory_array[offset]

def set_byte(offset: int, value: int):
    memory_array[offset] = value & 0xFF
"""
                init_file.write_text(init_content)
                
                # Create files for each possible low byte (0x00-0xFF)
                for file_addr in range(0x100):
                    file_path = dir_path / f"{file_addr:02x}.bin"
                    file_path.touch()
    
    def get_path_for_address(self, address: int) -> tuple[pathlib.Path, int]:
        """Convert a 16-bit address into directory path and offset"""
        if not 0 <= address <= self.MAX_ADDRESS:
            raise ValueError(f"Address 0x{address:04x} out of range")
        
        dir_addr = (address & self.DIRECTORY_MASK) >> 8
        file_addr = address & self.FILE_MASK
        
        dir_path = self.base_path / f"{dir_addr:02x}"
        file_path = dir_path / f"{file_addr:02x}.bin"
        
        return file_path, file_addr
    
    def read_word(self, address: int) -> bytes:
        """Read a word from the specified address"""
        file_path, offset = self.get_path_for_address(address)
        
        if address in self.memory_map:
            return self.memory_map[address].value
            
        try:
            with open(file_path, 'rb') as f:
                f.seek(offset)
                value = f.read(self.WORD_SIZE)
                if len(value) < self.WORD_SIZE:
                    value = value.ljust(self.WORD_SIZE, b'\x00')
                
                self.memory_map[address] = MemoryCell(value=value)
                return value
        except IOError:
            return b'\x00' * self.WORD_SIZE
    
    def write_word(self, address: int, value: bytes):
        """Write a word to the specified address"""
        if len(value) != self.WORD_SIZE:
            raise ValueError(f"Value must be {self.WORD_SIZE} bytes")
            
        file_path, offset = self.get_path_for_address(address)
        
        # Update memory map
        self.memory_map[address] = MemoryCell(value=value, dirty=True)
        
        # Write through to filesystem
        try:
            with open(file_path, 'r+b') as f:
                f.seek(offset)
                f.write(value)
        except IOError as e:
            raise IOError(f"Failed to write to address 0x{address:04x}: {e}")
    
    def flush(self):
        """Flush all dirty memory cells to disk"""
        for address, cell in self.memory_map.items():
            if cell.dirty:
                file_path, offset = self.get_path_for_address(address)
                with open(file_path, 'r+b') as f:
                    f.seek(offset)
                    f.write(cell.value)
                cell.dirty = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def __getitem__(self, address: int) -> bytes:
            return self.read_word(address)

    def __setitem__(self, address: int, value: bytes):
        self.write_word(address, value)
