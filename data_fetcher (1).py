import os
import time
import pandas as pd
import requests
from pathlib import Path
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
# CONFIGURATION (LOCK THESE VALUES)

MODE = "full"  
PILOT_SIZE = 10

IMAGE_SIZE = 224
ZOOM_LEVEL = 15
MAP_STYLE = "mapbox/satellite-v9"
BEARING = 0
PITCH = 0

# File paths
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Output directories
PILOT_DIR = Path("images/pilot")
TRAIN_DIR = Path("images/train")
TEST_DIR = Path("images/test")

# API settings
MAPBOX_TOKEN = "" # create account and add your own api here
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds


# CORE FUNCTIONS

def validate_coordinates(lat: float, lon: float) -> bool:
    """Check if coordinates are valid."""
    if pd.isna(lat) or pd.isna(lon):
        return False
    if not (-90 <= lat <= 90):
        return False
    if not (-180 <= lon <= 180):
        return False
    return True


def build_mapbox_url(lat: float, lon: float) -> str:
    """Construct Mapbox Static Images API URL."""
    url = (
        f"https://api.mapbox.com/styles/v1/{MAP_STYLE}/static/"
        f"{lon},{lat},{ZOOM_LEVEL},{BEARING},{PITCH}/"
        f"{IMAGE_SIZE}x{IMAGE_SIZE}@2x"
        f"?access_token={MAPBOX_TOKEN}"
    )
    return url


def download_image(url: str, output_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Download image with retry logic.
    Returns: (success: bool, error_message: Optional[str])
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True, None
            
            elif response.status_code == 401:
                return False, "Invalid API token"
            
            else:
                error_msg = f"HTTP {response.status_code}"
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return False, error_msg
                
        except requests.exceptions.Timeout:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            return False, "Timeout"
        
        except Exception as e:
            return False, str(e)
    
    return False, "Max retries exceeded"


def process_dataset(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    """Process a single dataset (train or test)."""
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Counters
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        # Get ID (try common column names)
        property_id = None
        for col in ['id', 'ID', 'property_id', 'PropertyID']:
            if col in row.index:
                property_id = row[col]
                break
        
        if property_id is None:
            print(f"[ERROR] Row {idx}: No ID column found")
            error_count += 1
            continue
        
        # Get coordinates (try common column names)
        lat = None
        lon = None
        
        for lat_col in ['latitude', 'Latitude', 'lat', 'Lat']:
            if lat_col in row.index:
                lat = row[lat_col]
                break
        
        for lon_col in ['longitude', 'Longitude', 'lon', 'Lon', 'long', 'Long']:
            if lon_col in row.index:
                lon = row[lon_col]
                break
        
        if lat is None or lon is None:
            print(f"[ERROR] id={property_id}: Missing coordinates")
            error_count += 1
            continue
        
        # Validate coordinates
        if not validate_coordinates(lat, lon):
            print(f"[ERROR] id={property_id}: Invalid coordinates ({lat}, {lon})")
            error_count += 1
            continue
        
        # Check if image already exists
        output_path = output_dir / f"{property_id}.png"
        if output_path.exists():
            print(f"[SKIP] id={property_id}: Already exists")
            skipped_count += 1
            continue
        
        # Build URL and download
        url = build_mapbox_url(lat, lon)
        success, error = download_image(url, output_path)
        
        if success:
            print(f"[OK] id={property_id}: Saved")
            success_count += 1
        else:
            print(f"[ERROR] id={property_id}: {error}")
            error_count += 1
        
        # Rate limiting (be nice to API)
        time.sleep(0.1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Success: {success_count}")
    print(f"⊘ Skipped: {skipped_count}")
    print(f"✗ Errors:  {error_count}")
    print(f"Total:     {len(df)}")
    print()



# MAIN EXECUTION

def main():
    """Main execution flow."""
    
    # Validate token
    if not MAPBOX_TOKEN:
        print("ERROR: MAPBOX_API_KEY environment variable not set!")
        print("Run: setx MAPBOX_API_KEY \"your_token_here\"")
        print("Then restart VS Code")
        return
    
    print("✓ Mapbox token found")
    print(f"✓ Mode: {MODE}")
    print()
    
    # Load datasets
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        print(f"Loaded train data: {len(train_df)} rows")
    except FileNotFoundError:
        print(f" ERROR: {TRAIN_FILE} not found!")
        return
    
    try:
        test_df = pd.read_csv(TEST_FILE)
        print(f"Loaded test data: {len(test_df)} rows")
    except FileNotFoundError:
        print(f"ERROR: {TEST_FILE} not found!")
        return
    
    # Execute based on mode
    if MODE == "pilot":
        print(f"\n PILOT MODE: Downloading {PILOT_SIZE} random training images")
        pilot_df = train_df.sample(n=min(PILOT_SIZE, len(train_df)), random_state=42)
        process_dataset(pilot_df, PILOT_DIR, "pilot")
        
        print("\n" + "="*60)
        print(" PILOT COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Check images/pilot/ folder")
        print("2. Open 10-20 images manually")
        print("3. If they look good, change MODE to 'full'")
        print("4. Run script again")
        
    elif MODE == "full":
        print("\n FULL MODE: Downloading all images")
        process_dataset(train_df, TRAIN_DIR, "train")
        process_dataset(test_df, TEST_DIR, "test")
        
        print("\n" + "="*60)
        print(" ALL DONE!")
        print("="*60)
        print("Your images are ready:")
        print(f"  - Train: images/train/")
        print(f"  - Test:  images/test/")
    
    else:
        print(f" ERROR: Invalid MODE '{MODE}'. Use 'pilot' or 'full'")


if __name__ == "__main__":
    main()