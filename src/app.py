import os
import pathlib
from typing import Dict, Optional, Union
from dataclasses import dataclass
import struct

@dataclass
class MemoryCell:
    """Represents a single addressable memory location"""
    value: bytes = b'\x00'  # Initialize with null byte
    
class VirtualMemoryFS:
    """
    Maps a hexadecimal word-addressed virtual memory to filesystem structure.
    Uses 16-bit addressing (0x0000 to 0xFFFF) split into:
    - Upper byte (0x00-0xFF): Directory address
    - Lower byte (0x00-0xFF): File address
    """
    WORD_SIZE = 2  # 16-bit addressing (two bytes)
    CELL_SIZE = 1  # 1 byte per cell
    BASE_DIR = "./app/"
    
    def __init__(self):
        """Initialize the virtual memory filesystem structure"""
        self.base_path = pathlib.Path(self.BASE_DIR)
        self._init_filesystem()
        self._memory_cache: Dict[int, MemoryCell] = {}
        
    def _init_filesystem(self):
        """Create the filesystem structure for virtual memory"""
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure (0x00-0xFF)
        for dir_addr in range(0x100):
            dir_path = self.base_path / f"{dir_addr:02x}"
            dir_path.mkdir(exist_ok=True)
            
            # Create __init__.py for the directory
            init_content = f"""
# Auto-generated __init__.py for virtual memory directory 0x{dir_addr:02x}
from dataclasses import dataclass, field
import array

@dataclass
class MemorySegment:
    data: array.array = field(default_factory=lambda: array.array('B', [0] * 256))
"""
            (dir_path / "__init__.py").write_text(init_content)
            
            # Create file structure (0x00-0xFF) within each directory
            for file_addr in range(0x100):
                file_path = dir_path / f"{file_addr:02x}"
                if not file_path.exists():
                    file_path.write_bytes(b'\x00')  # Initialize with null byte
                    
    def _address_to_path(self, address: int) -> pathlib.Path:
        """Convert a 16-bit address to filesystem path"""
        if not 0 <= address <= 0xFFFF:
            raise ValueError(f"Address {address:04x} out of range")
            
        dir_addr = (address >> 8) & 0xFF  # Upper byte
        file_addr = address & 0xFF        # Lower byte
        
        return self.base_path / f"{dir_addr:02x}" / f"{file_addr:02x}"
        
    def read(self, address: int) -> bytes:
        """Read a byte from the specified address"""
        if address in self._memory_cache:
            return self._memory_cache[address].value
            
        path = self._address_to_path(address)
        value = path.read_bytes()
        self._memory_cache[address] = MemoryCell(value)
        return value
        
    def write(self, address: int, value: bytes):
        """Write a byte to the specified address"""
        if len(value) != self.CELL_SIZE:
            raise ValueError(f"Value must be {self.CELL_SIZE} byte")
            
        path = self._address_to_path(address)
        path.write_bytes(value)
        self._memory_cache[address] = MemoryCell(value)
        
    def dump_segment(self, start_addr: int, length: int) -> bytes:
        """Dump a segment of memory"""
        result = bytearray()
        for addr in range(start_addr, start_addr + length):
            result.extend(self.read(addr))
        return bytes(result)

# Example usage
if __name__ == "__main__":
    vmem = VirtualMemoryFS()
    
    # Write some test values
    vmem.write(0x1234, b'\x42')
    vmem.write(0x1235, b'\xFF')
    
    # Read values back
    print(f"Value at 0x1234: {vmem.read(0x1234).hex()}")
    print(f"Value at 0x1235: {vmem.read(0x1235).hex()}")
    
    # Dump a memory segment
    print(f"Memory segment 0x1234-0x1236: {vmem.dump_segment(0x1234, 2).hex()}")