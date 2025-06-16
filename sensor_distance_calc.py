import math

# Johnson Criteria pixel requirements
pixel_reqs = {
    'detection': 2,
    'recognition': 8,
    'identification': 12.8,
}

# More realistic requirements (5x)
realistic_pixel_reqs = {k: v * 5 for k, v in pixel_reqs.items()}

# List of cameras with their focal length (mm) and pixel size (µm)
cameras = [
    {'name': 'DJI Mavic 3 Classic', 'f_mm': 24, 'pixel_size_um': 3.3},
    {'name': 'DJI Matrice 350 RTK', 'f_mm': 24, 'pixel_size_um': 4.4},
    {'name': 'DJI O4 Air Unit', 'f_mm': 14, 'pixel_size_um': 2.0},
    {'name': 'DJI O4 Air Unit Pro', 'f_mm': 14, 'pixel_size_um': 1.5},
    {'name': 'Puma AE (Mantis i45)', 'f_mm': 9, 'pixel_size_um': 1.6},  # pixel size approx.
    {'name': 'Caddx Ratel 2 (analog)', 'f_mm': 2.1, 'pixel_size_um': 5.6},
    {'name': 'Runcam Eagle 2 Pro (analog)', 'f_mm': 2.8, 'pixel_size_um': 5.6},
]

# Target sizes (meters)
target_sizes = {
    'human': 1.8,
    'vehicle': 5,
}

def calc_range_m(f_mm, target_m, pixel_size_um, pixels):
    """
    Calculate maximum range (meters) for a target of height target_m (m)
    given camera focal length f_mm (mm), pixel size pixel_size_um (µm),
    and required pixels on target.
    """
    p_mm = pixel_size_um / 1000.0  # convert µm to mm
    return (f_mm * target_m) / (p_mm * pixels)

print(f"{'Camera':<30} {'Target':<10} {'D (km)':>8} {'R (km)':>8} {'I (km)':>8}")
print("-" * 70)

for cam in cameras:
    for target_name, H in target_sizes.items():
        D = calc_range_m(cam['f_mm'], H, cam['pixel_size_um'], realistic_pixel_reqs['detection']) / 1000.0
        R = calc_range_m(cam['f_mm'], H, cam['pixel_size_um'], realistic_pixel_reqs['recognition']) / 1000.0
        I = calc_range_m(cam['f_mm'], H, cam['pixel_size_um'], realistic_pixel_reqs['identification']) / 1000.0
        print(f"{cam['name']:<30} {target_name:<10} {D:>8.2f} {R:>8.2f} {I:>8.2f}")
