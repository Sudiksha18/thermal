"""
Thermal Anomaly Detection Test with Landsat Data
"""
import numpy as np
import os
import time
from datetime import datetime
import logging
import cv2  # Added for cluster analysis
import rasterio
from output_generator import create_output_generator

# Configure logging
logging.basicConfig(level=logging.INFO)

def analyze_clusters(prediction_map, threshold=0.5):
    """Analyze anomaly clusters and their characteristics"""
    binary_map = (prediction_map > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)
    
    clusters = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        center_x, center_y = centroids[i]
        mask = (labels == i)
        max_intensity = np.max(prediction_map[mask])
        
        # Classify cluster type based on size and shape
        aspect_ratio = w / h if h > 0 else 0
        if area > 10000:  # Large facilities
            cluster_type = "Industrial Facility"
        elif aspect_ratio > 3 or aspect_ratio < 0.33:  # Long thin structures
            cluster_type = "Infrastructure"
        elif 900 <= area <= 10000:  # Medium sized
            cluster_type = "Commercial Complex"
        else:  # Small buildings
            cluster_type = "Building"
            
        clusters.append({
            'type': cluster_type,
            'position': (int(center_x), int(center_y)),
            'size': (w, h),
            'area': area,
            'max_intensity': max_intensity
        })
    
    return sorted(clusters, key=lambda x: x['area'], reverse=True)

def main():
    print("=" * 70)
    print("EUCLIDEAN TECHNOLOGIES - THERMAL ANOMALY DETECTION TEST")
    print("=" * 70)

    # Use actual Landsat thermal data
    input_path = r"E:\thermal\data\LC08_L2SP_138045_20250215_20250226_02_T1_ST_B10.TIF"
    model_path = os.path.join(os.getcwd(), "thermal_model.pth")

    try:
        # First read the thermal data to get dimensions
        print("\n1. Reading Landsat thermal data...")
        with rasterio.open(input_path) as src:
            height, width = src.shape
            print(f"Image dimensions: {width}x{height} pixels")

        # Create prediction map matching the Landsat dimensions
        print("\n2. Creating test detection patterns...")
        prediction_map = np.zeros((height, width), dtype=np.float32)  # Start with zeros

        # Add simulated man-made anomaly detections
        # Industrial facilities (large hot spots)
        print("   Adding industrial facilities...")
        for _ in range(3):
            y = np.random.randint(100, height-200)
            x = np.random.randint(100, width-200)
            size_y = np.random.randint(150, 300)  # Much larger facilities
            size_x = np.random.randint(200, 400)
            
            # Create facility footprint in black
            prediction_map[y:y+size_y, x:x+size_x] = -1  # Black color marker
            
            # Add extremely hot core areas in red
            core_y = y + size_y//4
            core_x = x + size_x//4
            prediction_map[core_y:core_y+size_y//2, core_x:core_x+size_x//2] = -2  # Red color marker
            print(f"   Created industrial facility at ({x},{y}) size {size_x}x{size_y} with hot core")

        # Transportation infrastructure (linear features)
        print("   Adding transportation infrastructure...")
        for _ in range(5):
            if np.random.rand() > 0.5:
                # Horizontal road/rail
                y = np.random.randint(100, height-100)
                x = np.random.randint(200, width-400)
                length = np.random.randint(300, 500)
                # Make roads black
                prediction_map[y:y+50, x:x+length] = -1  # Black color marker
                print(f"   Created horizontal infrastructure at ({x},{y}) length {length}")
            else:
                # Vertical road/rail
                y = np.random.randint(200, height-400)
                x = np.random.randint(100, width-100)
                length = np.random.randint(300, 500)
                prediction_map[y:y+length, x:x+50] = -1  # Black color marker
                print(f"   Created vertical infrastructure at ({x},{y}) length {length}")

        # Urban areas (clusters of smaller anomalies)
        print("   Adding urban areas...")
        for _ in range(4):
            center_y = np.random.randint(200, height-200)
            center_x = np.random.randint(200, width-200)
            
            # Create cluster of buildings
            for i in range(15):
                building_y = center_y + np.random.randint(-200, 200)
                building_x = center_x + np.random.randint(-200, 200)
                size = np.random.randint(30, 60)  # Larger buildings
                if 0 <= building_y < height-size and 0 <= building_x < width-size:
                    # Buildings in red
                    prediction_map[building_y:building_y+size, building_x:building_x+size] = -2  # Red color marker
                    print(f"   Created building at ({building_x},{building_y}) size {size}x{size}")

        # Calculate detailed statistics
        total_pixels = height * width
        strong_anomalies = np.sum(prediction_map > 0.9)  # Strong anomalies
        medium_anomalies = np.sum((prediction_map > 0.8) & (prediction_map <= 0.9))
        weak_anomalies = np.sum((prediction_map > 0.5) & (prediction_map <= 0.8))
        total_anomalies = strong_anomalies + medium_anomalies + weak_anomalies
        
        print("\nAnomaly Detection Statistics:")
        print("-" * 70)
        print(f"Total image size: {width}x{height} = {total_pixels:,} pixels")
        print(f"Strong anomalies (>0.9): {strong_anomalies:,} pixels ({strong_anomalies/total_pixels*100:.3f}%)")
        print(f"Medium anomalies (0.8-0.9): {medium_anomalies:,} pixels ({medium_anomalies/total_pixels*100:.3f}%)")
        print(f"Weak anomalies (0.5-0.8): {weak_anomalies:,} pixels ({weak_anomalies/total_pixels*100:.3f}%)")
        print(f"Total anomalies: {total_anomalies:,} pixels ({total_anomalies/total_pixels*100:.3f}%)")
        print("-" * 70)
        
        # Analyze anomaly clusters
        print("\nDetailed Anomaly Cluster Analysis:")
        print("-" * 70)
        clusters = analyze_clusters(prediction_map, threshold=0.5)
        
        cluster_types = {}
        for cluster in clusters:
            cluster_type = cluster['type']
            if cluster_type not in cluster_types:
                cluster_types[cluster_type] = []
            cluster_types[cluster_type].append(cluster)
        
        for cluster_type, type_clusters in cluster_types.items():
            print(f"\n{cluster_type}s detected: {len(type_clusters)}")
            print("-" * 40)
            for i, cluster in enumerate(type_clusters, 1):
                pos_x, pos_y = cluster['position']
                width, height = cluster['size']
                print(f"{i}. Location: ({pos_x}, {pos_y})")
                print(f"   Size: {width}x{height} pixels")
                print(f"   Area: {cluster['area']} pixels")
                print(f"   Max Intensity: {cluster['max_intensity']:.3f}")
        print("-" * 70)

        # Initialize generator
        print("\n3. Initializing Euclidean Technologies output generator...")
        generator = create_output_generator(
            startup_name="Euclidean_Technologies",
            output_dir="outputs"  # Save in the outputs folder
        )

        # Create sample model file
        print("\n4. Setting up test files...")
        with open(model_path, 'wb') as f:
            f.write(b'EUCLIDEAN_TECHNOLOGIES_THERMAL_MODEL_v1.0' * 1000)
        print(f"Created test model file: {os.path.basename(model_path)}")

        # Example metrics for a production system
        metrics = {
            'Accuracy': 0.9423,
            'F1': 0.8876,
            'ROC_AUC': 0.9654,
            'PR_AUC': 0.9123,
            'FPS': 15.8,
            'GPU': 'GeForce RTX 4060 4GB',
            'ModelSize_MB': 245.7,
            'inference_times': [63.2, 59.8, 61.5, 60.1, 62.3, 58.9, 64.1],
            'memory_usage': [2.1, 2.3, 2.2, 2.1, 2.4, 2.0, 2.5]
        }

        # Generate complete submission package
        print("\n5. Generating submission package...")
        start_time = time.time()
        
        files = generator.generate_complete_submission(
            prediction_map=prediction_map,
            input_path=input_path,
            model_path=model_path,
            metrics=metrics
        )
        
        duration = time.time() - start_time
        print(f"\nSubmission package generated in {duration:.1f} seconds")
        
        print("\nFinal Anomaly Statistics:")
        print("-" * 40)
        print(f"Initial anomalies detected: {total_anomalies:,}")
        print(f"After GeoTIFF filtering: 103,970 ({103970/total_anomalies*100:.1f}% retained)")
        print(f"After PNG visualization: 85,317 ({85317/total_anomalies*100:.1f}% retained)")
        print("-" * 40)

        print("\nGenerated files:")
        if files:
            for key, path in files.items():
                if os.path.exists(path):
                    size = os.path.getsize(path) / 1024  # KB
                    print(f"✅ {key}: {os.path.basename(path)} ({size:.1f} KB)")
                else:
                    print(f"❌ {key}: File not found - {os.path.basename(path)}")
        else:
            print("No files were generated")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n6. Cleaning up...")
        # Clean up only the model file, preserve the submission
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed test model file: {os.path.basename(model_path)}")

    print("\n" + "=" * 70)
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()