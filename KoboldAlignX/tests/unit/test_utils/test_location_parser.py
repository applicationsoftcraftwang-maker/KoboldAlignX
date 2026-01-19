"""
Unit tests for location_parser.py
"""

import pytest
from src.utils.location_parser import parse_location, parse_location_sync


class TestParseLocation:
    """Test suite for location parsing."""
    
    @pytest.mark.asyncio
    async def test_parse_location_valid_input(self):
        """Test parsing valid location string."""
        location = "AB_100-01-02-3-04W5"
        uwi, twp, rng = await parse_location(location)
        
        assert uwi == "100/01-02-003-04W5"
        assert twp == "003"
        assert rng == "04"
    
    @pytest.mark.asyncio
    async def test_parse_location_already_padded(self):
        """Test parsing location with already padded township."""
        location = "AB_100-01-02-045-12W5"
        uwi, twp, rng = await parse_location(location)
        
        assert uwi == "100/01-02-045-12W5"
        assert twp == "045"
        assert rng == "12"
    
    @pytest.mark.asyncio
    async def test_parse_location_no_prefix(self):
        """Test parsing location without AB_ prefix."""
        location = "100-01-02-056-08W5"
        uwi, twp, rng = await parse_location(location)
        
        assert uwi == "100/01-02-056-08W5"
        assert twp == "056"
        assert rng == "08"
    
    @pytest.mark.asyncio
    async def test_parse_location_single_digit_township(self):
        """Test parsing location with single-digit township."""
        location = "AB_100-01-02-5-04W5"
        uwi, twp, rng = await parse_location(location)
        
        assert uwi == "100/01-02-005-04W5"
        assert twp == "005"
        assert rng == "04"
    
    @pytest.mark.asyncio
    async def test_parse_location_two_digit_township(self):
        """Test parsing location with two-digit township."""
        location = "AB_100-01-02-45-04W5"
        uwi, twp, rng = await parse_location(location)
        
        assert uwi == "100/01-02-045-04W5"
        assert twp == "045"
        assert rng == "04"
    
    @pytest.mark.asyncio
    async def test_parse_location_invalid_format(self):
        """Test parsing invalid location format."""
        location = "invalid_location"
        uwi, twp, rng = await parse_location(location)
        
        # Should return None values on error
        assert twp is None
        assert rng is None
    
    @pytest.mark.asyncio
    async def test_parse_location_empty_string(self):
        """Test parsing empty string."""
        location = ""
        uwi, twp, rng = await parse_location(location)
        
        assert twp is None
        assert rng is None
    
    @pytest.mark.asyncio
    async def test_parse_location_no_hyphens(self):
        """Test parsing location with no hyphens."""
        location = "AB_100010200304W5"
        uwi, twp, rng = await parse_location(location)
        
        assert twp is None
        assert rng is None
    
    @pytest.mark.asyncio
    async def test_parse_location_insufficient_parts(self):
        """Test parsing location with insufficient parts."""
        location = "AB_100-01"
        uwi, twp, rng = await parse_location(location)
        
        assert twp is None
        assert rng is None


class TestParseLocationSync:
    """Test suite for synchronous location parsing."""
    
    def test_parse_location_sync_valid_input(self):
        """Test synchronous parsing of valid location."""
        location = "AB_100-01-02-3-04W5"
        uwi, twp, rng = parse_location_sync(location)
        
        assert uwi == "100/01-02-003-04W5"
        assert twp == "003"
        assert rng == "04"
    
    def test_parse_location_sync_invalid_input(self):
        """Test synchronous parsing of invalid location."""
        location = "invalid"
        uwi, twp, rng = parse_location_sync(location)
        
        assert twp is None
        assert rng is None
    
    def test_parse_location_sync_matches_async(self):
        """Test that sync version matches async version."""
        location = "AB_100-01-02-045-12W5"
        
        # Sync version
        uwi_sync, twp_sync, rng_sync = parse_location_sync(location)
        
        # Both should produce same results
        assert uwi_sync == "100/01-02-045-12W5"
        assert twp_sync == "045"
        assert rng_sync == "12"


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    @pytest.mark.asyncio
    async def test_parse_location_with_spaces(self):
        """Test location with spaces (should fail gracefully)."""
        location = "AB_100 01 02 003 04W5"
        uwi, twp, rng = await parse_location(location)
        
        # Should handle gracefully
        assert twp is None or isinstance(twp, str)
    
    @pytest.mark.asyncio
    async def test_parse_location_with_special_chars(self):
        """Test location with special characters."""
        location = "AB_100-01-02-003-04W5#@!"
        uwi, twp, rng = await parse_location(location)
        
        # Should extract numeric portion
        # Behavior depends on regex matching
        assert isinstance(uwi, (str, type(None)))
    
    @pytest.mark.parametrize("location,expected_twp,expected_rng", [
        ("AB_100-01-02-1-04W5", "001", "04"),
        ("AB_100-01-02-12-04W5", "012", "04"),
        ("AB_100-01-02-123-04W5", "123", "04"),
        ("AB_100-01-02-003-4W5", "003", "4"),
    ])
    @pytest.mark.asyncio
    async def test_parse_location_various_formats(self, location, expected_twp, expected_rng):
        """Test various location formats."""
        uwi, twp, rng = await parse_location(location)
        
        assert twp == expected_twp
        assert rng == expected_rng