import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math

def create_deepfake_comparison():
    width, height = 800, 400
    
    # Create a new image with gradient background
    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw gradient background
    for x in range(width):
        r = int(114 - (114 - 76) * x / width)
        g = int(9 - (9 - 201) * x / width)
        b = int(183 - (183 - 240) * x / width)
        for y in range(height):
            draw.point((x, y), fill=(r, g, b))
    
    # Draw divider
    for y in range(50, height - 50):
        draw.rectangle([(width//2 - 2, y), (width//2 + 2, y)], fill=(255, 255, 255, 50))
    
    # Draw labels
    draw.text((width//4, 30), "REAL", fill=(255, 255, 255), anchor="mm")
    draw.text((3*width//4, 30), "DEEPFAKE", fill=(255, 255, 255), anchor="mm")
    
    # Draw face circles
    draw.ellipse((width//4 - 100, height//2 - 100, width//4 + 100, height//2 + 100), outline=(255, 255, 255), width=3)
    draw.ellipse((3*width//4 - 100, height//2 - 100, 3*width//4 + 100, height//2 + 100), outline=(255, 255, 255), width=3)
    
    # Draw real face features
    # Eyes
    draw.ellipse((width//4 - 60, height//2 - 40, width//4 - 20, height//2 - 10), fill=(255, 255, 255))
    draw.ellipse((width//4 + 20, height//2 - 40, width//4 + 60, height//2 - 10), fill=(255, 255, 255))
    
    # Nose
    draw.line([(width//4, height//2 - 10), (width//4, height//2 + 20), (width//4 + 15, height//2 + 25)], fill=(255, 255, 255), width=3)
    
    # Mouth
    draw.arc([(width//4 - 40, height//2 + 35), (width//4 + 40, height//2 + 65)], 0, 180, fill=(255, 255, 255), width=3)
    
    # Draw deepfake face features with distortions
    # Eyes (slightly off)
    draw.ellipse((3*width//4 - 65, height//2 - 50, 3*width//4 - 15, height//2 - 10), fill=(255, 255, 255))
    draw.ellipse((3*width//4 + 15, height//2 - 45, 3*width//4 + 65, height//2 - 5), fill=(255, 255, 255))
    
    # Nose (misaligned)
    draw.line([(3*width//4, height//2), (3*width//4 - 10, height//2 + 30), (3*width//4 + 15, height//2 + 30)], fill=(255, 255, 255), width=3)
    
    # Mouth (distorted)
    draw.arc([(3*width//4 - 50, height//2 + 45), (3*width//4 + 50, height//2 + 75)], 20, 160, fill=(255, 255, 255), width=3)
    
    # Add glitch effects
    for _ in range(8):
        glitch_x = 3*width//4 - 100 + random.randint(0, 200)
        glitch_y = height//2 - 100 + random.randint(0, 200)
        glitch_width = 20 + random.randint(0, 80)
        glitch_height = 3 + random.randint(0, 10)
        
        r = random.randint(200, 255)
        g = random.randint(0, 50)
        b = random.randint(100, 150)
        a = random.randint(30, 80)
        
        for x in range(glitch_width):
            for y in range(glitch_height):
                if 0 <= glitch_x + x < width and 0 <= glitch_y + y < height:
                    r_curr, g_curr, b_curr = img.getpixel((glitch_x + x, glitch_y + y))
                    r_new = min(255, r_curr + (r - r_curr) * a // 255)
                    g_new = min(255, g_curr + (g - g_curr) * a // 255)
                    b_new = min(255, b_curr + (b - b_curr) * a // 255)
                    img.putpixel((glitch_x + x, glitch_y + y), (r_new, g_new, b_new))
    
    # Add pixel noise
    for _ in range(200):
        noise_x = 3*width//4 - 100 + random.randint(0, 200)
        noise_y = height//2 - 100 + random.randint(0, 200)
        noise_size = 1 + random.randint(0, 3)
        
        r = random.randint(0, 50)
        g = random.randint(200, 255)
        b = random.randint(200, 255)
        a = random.randint(50, 100)
        
        for x in range(noise_size):
            for y in range(noise_size):
                if 0 <= noise_x + x < width and 0 <= noise_y + y < height:
                    r_curr, g_curr, b_curr = img.getpixel((noise_x + x, noise_y + y))
                    r_new = min(255, r_curr + (r - r_curr) * a // 255)
                    g_new = min(255, g_curr + (g - g_curr) * a // 255)
                    b_new = min(255, b_curr + (b - b_curr) * a // 255)
                    img.putpixel((noise_x + x, noise_y + y), (r_new, g_new, b_new))
    
    # Add footer text
    draw.text((width//2, height - 30), "Real vs Deepfake Comparison", fill=(255, 255, 255), anchor="mm")
    
    img.save("deepfake-comparison.png")
    print("Created deepfake comparison image")

def create_ai_detection():
    width, height = 800, 400
    
    # Create a new image with gradient background
    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw gradient background
    for x in range(width):
        r = int(67 - (67 - 76) * x / width)
        g = int(97 - (97 - 201) * x / width)
        b = int(238 - (238 - 240) * x / width)
        for y in range(height):
            draw.point((x, y), fill=(r, g, b))
    
    # Draw grid pattern
    for x in range(0, width, 20):
        for y in range(0, height, 20):
            draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 25), width=1)
            draw.line([(0, y), (width, y)], fill=(255, 255, 255, 25), width=1)
    
    # Draw title
    draw.text((width//2, 30), "DeepGuard AI Detection Process", fill=(255, 255, 255), anchor="mm")
    
    # Define steps positions
    steps_x = [int(width * 0.15), int(width * 0.38), int(width * 0.62), int(width * 0.85)]
    steps_y = height // 2
    radius = 50
    
    # Draw connecting lines
    for i in range(len(steps_x) - 1):
        for dash_start in range(steps_x[i] + radius, steps_x[i+1] - radius, 13):
            draw.line([(dash_start, steps_y), (dash_start + 5, steps_y)], fill=(255, 255, 255, 180), width=4)
    
    # Process steps and icons
    steps = ["Frame\nExtraction", "Face\nDetection", "Neural\nAnalysis", "Result\nGeneration"]
    icons = ["📹", "👤", "🧠", "📊"]  # Unicode icons
    
    # Draw steps
    for i, (x, step, icon) in enumerate(zip(steps_x, steps, icons)):
        # Circle with gradient effect
        for r in range(radius, 0, -1):
            alpha = int(75 + (180 - 75) * (radius - r) / radius)
            draw.ellipse((x - r, steps_y - r, x + r, steps_y + r), fill=(255, 255, 255, alpha))
        
        # Step number
        draw.text((x, steps_y - radius - 15), f"Step {i+1}", fill=(255, 255, 255), anchor="mm")
        
        # Icon (simplified as text)
        draw.text((x, steps_y), icon, fill=(67, 97, 238), anchor="mm")
        
        # Label
        lines = step.split('\n')
        for j, line in enumerate(lines):
            draw.text((x, steps_y + radius + 25 + j * 20), line, fill=(255, 255, 255), anchor="mm")
    
    img.save("ai-detection.png")
    print("Created AI detection process image")

def create_pattern_background():
    width, height = 500, 500
    
    # Create a new image with dark background
    img = Image.new('RGB', (width, height), (26, 26, 46))
    draw = ImageDraw.Draw(img)
    
    # Create nodes
    node_count = 30
    nodes = []
    
    for _ in range(node_count):
        nodes.append({
            'x': random.randint(0, width),
            'y': random.randint(0, height),
            'radius': 1 + random.random() * 3
        })
    
    # Draw connections
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dx = nodes[i]['x'] - nodes[j]['x']
            dy = nodes[i]['y'] - nodes[j]['y']
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < 100:
                draw.line([(nodes[i]['x'], nodes[i]['y']), (nodes[j]['x'], nodes[j]['y'])], 
                          fill=(255, 255, 255, 25), width=1)
    
    # Draw nodes
    for node in nodes:
        r = int(node['radius'])
        draw.ellipse((node['x'] - r, node['y'] - r, node['x'] + r, node['y'] + r), 
                    fill=(255, 255, 255, 128))
    
    img.save("pattern-bg.png")
    print("Created pattern background image")

if __name__ == "__main__":
    print("Generating images...")
    create_deepfake_comparison()
    create_ai_detection()
    create_pattern_background()
    print("All images generated successfully!") 