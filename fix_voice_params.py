"""
🎯 VOICE CLONE PARAMETER FIXER
==============================

This script fixes the critical cfg_weight parameter issue
and provides optimized settings for better voice similarity.

PROBLEM FOUND:
- Your cfg_weight is set to 4.0-4.5
- ChatterBox recommends 0.2-1.0 (default 0.5)
- Too high cfg_weight causes:
  • Distorted/unnatural speech
  • Over-fitting to reference
  • Loss of natural prosody

SOLUTION:
- Use cfg_weight between 0.3-0.7
- Use single best sample (sample_012.wav)
- Match exaggeration to speaking style
"""

import os
import sys
import json
from pathlib import Path

# The CORRECT parameter ranges for ChatterBox TTS
# Based on official documentation and research

CORRECT_PARAMS = {
    "cfg_weight": {
        "recommended_range": (0.2, 1.0),
        "default": 0.5,
        "for_similarity": 0.6,  # Slightly higher for better voice match
        "for_naturalness": 0.4,  # Slightly lower for more natural flow
        "for_cross_language": 0.0,  # Set to 0 for cross-language
        "YOUR_CURRENT": "4.0-4.5 (WAY TOO HIGH!)"
    },
    "exaggeration": {
        "recommended_range": (0.25, 2.0),
        "default": 0.5,
        "neutral": 0.4,  # Slightly under for calm speech
        "expressive": 0.7,  # For dramatic/excited
        "monotone": 0.25,  # Very flat
        "YOUR_CURRENT": "0.2-0.8 (OK)"
    },
    "temperature": {
        "recommended_range": (0.05, 5.0),
        "default": 0.8,
        "consistent": 0.5,  # Lower = more consistent
        "varied": 1.0,  # Higher = more varied
        "YOUR_CURRENT": "0.5-0.7 (OK)"
    }
}

# OPTIMIZED emotion presets based on research
OPTIMIZED_PRESETS = {
    "neutral": {
        "exaggeration": 0.4,       # Slightly below neutral
        "temperature": 0.6,        # Consistent
        "cfg_weight": 0.6,         # Good balance of similarity and naturalness
    },
    "excited": {
        "exaggeration": 0.7,       # More expressive
        "temperature": 0.75,       # Slightly more varied
        "cfg_weight": 0.5,         # Standard - don't force similarity
    },
    "calm": {
        "exaggeration": 0.3,       # Less expressive
        "temperature": 0.5,        # Very consistent
        "cfg_weight": 0.7,         # Higher similarity for calm voice
    },
    "dramatic": {
        "exaggeration": 0.9,       # Very expressive
        "temperature": 0.8,        # More varied
        "cfg_weight": 0.4,         # Lower to allow natural expression
    },
    "conversational": {
        "exaggeration": 0.5,       # Natural
        "temperature": 0.7,        # Natural variation
        "cfg_weight": 0.5,         # Balanced
    },
    "storytelling": {
        "exaggeration": 0.6,       # Moderately expressive
        "temperature": 0.7,        # Natural variation
        "cfg_weight": 0.5,         # Balanced
    }
}


def print_analysis():
    """Print detailed analysis of the parameter issue"""
    
    print("=" * 70)
    print("🔬 VOICE CLONE PARAMETER ANALYSIS")
    print("=" * 70)
    
    print("\n❌ CRITICAL PROBLEM FOUND:")
    print("-" * 70)
    print("""
Your myvoiceclone.py uses cfg_weight = 4.0-4.5

This is WRONG! ChatterBox documentation says:
• cfg_weight range: 0.2-1.0
• Default: 0.5
• Higher values = closer to reference BUT can cause artifacts

Setting cfg_weight to 4.0+ causes:
• Over-exaggerated voice characteristics
• Unnatural/robotic speech patterns
• Loss of natural prosody
• Potential audio artifacts
""")
    
    print("\n✅ CORRECT PARAMETERS (from ChatterBox docs):")
    print("-" * 70)
    
    for param, info in CORRECT_PARAMS.items():
        print(f"\n{param}:")
        print(f"  Range: {info['recommended_range']}")
        print(f"  Default: {info['default']}")
        print(f"  Your current: {info['YOUR_CURRENT']}")
    
    print("\n" + "=" * 70)
    print("🛠️ RECOMMENDED FIX:")
    print("=" * 70)
    print("""
Replace your emotion_presets in myvoiceclone.py with these:
""")
    
    print(json.dumps(OPTIMIZED_PRESETS, indent=4))
    
    print("""
Or run this script with --apply to automatically fix it.
""")


def create_patch():
    """Create the code patch to fix the parameters"""
    
    old_code = '''        self.emotion_presets = {
            "neutral": {
                "exaggeration": 0.3,       # Low = stay close to original voice
                "temperature": 0.6,        # Lower = more consistent, less random
                "cfg_weight": 4.0,         # HIGH = strongly match your voice samples
            },
            "excited": {
                "exaggeration": 0.6,
                "temperature": 0.7,
                "cfg_weight": 4.0,
            },
            "calm": {
                "exaggeration": 0.2,
                "temperature": 0.5,
                "cfg_weight": 4.5,
            },
            "dramatic": {
                "exaggeration": 0.8,
                "temperature": 0.7,
                "cfg_weight": 3.5,
            },
            "conversational": {
                "exaggeration": 0.4,
                "temperature": 0.6,
                "cfg_weight": 4.0,
            },
            "storytelling": {
                "exaggeration": 0.5,
                "temperature": 0.65,
                "cfg_weight": 4.0,
            }
        }'''
    
    new_code = '''        # FIXED: cfg_weight should be 0.2-1.0, NOT 4.0!
        # Higher cfg_weight = more similar to reference but can cause artifacts
        # Optimal range: 0.4-0.7 for best similarity + naturalness
        self.emotion_presets = {
            "neutral": {
                "exaggeration": 0.4,       # Slightly below neutral for natural sound
                "temperature": 0.6,        # Lower = more consistent
                "cfg_weight": 0.6,         # FIXED: Was 4.0, now 0.6 for similarity
            },
            "excited": {
                "exaggeration": 0.7,       # Higher for expressive speech
                "temperature": 0.75,       # Slightly more varied
                "cfg_weight": 0.5,         # FIXED: Was 4.0, now balanced
            },
            "calm": {
                "exaggeration": 0.3,       # Lower for calm delivery
                "temperature": 0.5,        # Very consistent
                "cfg_weight": 0.7,         # FIXED: Was 4.5, higher for similarity
            },
            "dramatic": {
                "exaggeration": 0.9,       # High for dramatic effect
                "temperature": 0.8,        # More varied
                "cfg_weight": 0.4,         # FIXED: Was 3.5, lower to allow expression
            },
            "conversational": {
                "exaggeration": 0.5,       # Natural speaking
                "temperature": 0.7,        # Natural variation
                "cfg_weight": 0.5,         # FIXED: Was 4.0, balanced
            },
            "storytelling": {
                "exaggeration": 0.6,       # Moderately expressive
                "temperature": 0.7,        # Natural variation
                "cfg_weight": 0.5,         # FIXED: Was 4.0, balanced
            }
        }'''
    
    return old_code, new_code


def apply_fix(myvoiceclone_path: str):
    """Apply the fix to myvoiceclone.py"""
    
    with open(myvoiceclone_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_code, new_code = create_patch()
    
    if old_code not in content:
        print("⚠️ Could not find the exact code to replace.")
        print("   The file may have been modified.")
        return False
    
    new_content = content.replace(old_code, new_code)
    
    # Backup original
    backup_path = myvoiceclone_path + ".backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📁 Backup saved to: {backup_path}")
    
    # Write fixed version
    with open(myvoiceclone_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Fixed parameters in: {myvoiceclone_path}")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix voice clone parameters")
    parser.add_argument("--apply", action="store_true",
                       help="Apply the fix to myvoiceclone.py")
    parser.add_argument("--file", default="myvoiceclone.py",
                       help="Path to myvoiceclone.py")
    args = parser.parse_args()
    
    print_analysis()
    
    if args.apply:
        myvoiceclone_path = Path(__file__).parent / args.file
        
        if not myvoiceclone_path.exists():
            print(f"❌ File not found: {myvoiceclone_path}")
            return
        
        print("\n" + "=" * 70)
        print("🔧 APPLYING FIX...")
        print("=" * 70)
        
        if apply_fix(str(myvoiceclone_path)):
            print("\n✅ Fix applied successfully!")
            print("\nNext steps:")
            print("1. Run myvoiceclone.py")
            print("2. Use 'sample_012.wav' as your reference")
            print("3. Test with different emotion presets")
        else:
            print("\n❌ Fix could not be applied automatically.")
            print("   Please manually update the cfg_weight values.")
    else:
        print("\n💡 Run with --apply to automatically fix the file:")
        print(f"   python {Path(__file__).name} --apply")


if __name__ == "__main__":
    main()
