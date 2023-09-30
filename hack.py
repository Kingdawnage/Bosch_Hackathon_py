from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the development data
df = pd.read_csv("./DevelopmentData.xlsx - Sheet1.csv")

## Data preprocessing
# Convert the distances to meters
df['FirstObjectDistance_X'] = df['FirstObjectDistance_X'] / 128
df['FirstObjectDistance_Y'] = df['FirstObjectDistance_Y'] / 128

df['SecondObjectDistance_X'] = df['SecondObjectDistance_X'] / 128
df['SecondObjectDistance_Y'] = df['SecondObjectDistance_Y'] / 128

df['ThirdObjectDistance_X'] = df['ThirdObjectDistance_X'] / 128
df['ThirdObjectDistance_Y'] = df['ThirdObjectDistance_Y'] / 128

df['FourthObjectDistance_X'] = df['FourthObjectDistance_X'] / 128
df['FourthObjectDistance_Y'] = df['FourthObjectDistance_Y'] / 128

# Convert vehicle speed to m/s
df['VehicleSpeed'] = df['VehicleSpeed'] / 256  

# Convert object speeds to m/s
df['FirstObjectSpeed_X'] = df['FirstObjectSpeed_X'] / 256
df['FirstObjectSpeed_Y'] = df['FirstObjectSpeed_Y'] / 256

df['SecondObjectSpeed_X'] = df['SecondObjectSpeed_X'] / 256
df['SecondObjectSpeed_Y'] = df['SecondObjectSpeed_Y'] / 256

df['ThirdObjectSpeed_X'] = df['ThirdObjectSpeed_X'] / 256
df['ThirdObjectSpeed_Y'] = df['ThirdObjectSpeed_Y'] / 256

df['FourthObjectSpeed_X'] = df['FourthObjectSpeed_X'] / 256
df['FourthObjectSpeed_Y'] = df['FourthObjectSpeed_Y'] / 256

# Initialize necessary variables for calulating the kinematics of the vehicle and the objects
car_x, car_y = 0, 0
car_pos_x = []
car_pos_y = []
car_pos = []
object_pos1 = []
object_pos2 = []
object_pos3 = []
object_pos4 = []
car_acceleration_values = []

# Define and calculate the boundaries of the vehicle
car_width = 3
car_length = 1
car_size = 0.8 # For the size of the car in the time plot

# Define the breaking distance threshold etc
brake_thresh = 2
prev_velocity = df['VehicleSpeed'].iloc[0]
prev_timestamp = df['Timestamp'].iloc[0]

# Calculate the position of the car and the position of the objects 
# and implement the proximity ABS function
for index, row in df.iterrows():
    car_velocity = row['VehicleSpeed'] / 256
    car_yaw_rate = row['YawRate']
    time = row['Timestamp']
    delta_x = car_velocity * np.cos(car_yaw_rate) * time
    delta_y = car_velocity * np.sin(car_yaw_rate) * time
    
    car_x += delta_x
    car_y += delta_y
    
    car_pos_x.append(car_x)
    car_pos_y.append(car_y)
    car_pos.append((car_x, car_y, time))
    object_pos1.append((row['FirstObjectDistance_X'], row['FirstObjectDistance_Y'], time))
    object_pos2.append((row['SecondObjectDistance_X'], row['SecondObjectDistance_Y'], time))
    object_pos3.append((row['ThirdObjectDistance_X'], row['ThirdObjectDistance_Y'], time))
    object_pos4.append((row['FourthObjectDistance_X'], row['FourthObjectDistance_Y'], time))

    
    # Check the distance of objects to the boundary of the car
    object_distances_x = [row['FirstObjectDistance_X'],
                          row['SecondObjectDistance_X'],
                          row['ThirdObjectDistance_X'],
                          row['FourthObjectDistance_X']
                          ]
    object_distances_y = [row['FirstObjectDistance_Y'],
                          row['SecondObjectDistance_Y'],
                          row['ThirdObjectDistance_Y'],
                          row['FourthObjectDistance_Y']
                          ]

    # Calculate acceleration based on the change in velocity and time
    acceleration = (car_velocity - prev_velocity) / (time - prev_timestamp)

    brake_condition_met_x = (abs(car_x - object_distances_x[0]) < brake_thresh).any()
    brake_condition_met_y = (abs(car_y - object_distances_y[0]) < brake_thresh).any()

    if brake_condition_met_x or brake_condition_met_y:
        acceleration *= 0.85
      
    car_acceleration_values.append(acceleration)
    
object_pos = [object_distances_x, object_distances_y]
car_pos1 = [car_pos_x, car_pos_y]    

# Convert the position data into a numpy array
car_pos = np.array(car_pos)
object_pos = np.array(object_pos)
object_pos1 = np.array(object_pos1)
object_pos2 = np.array(object_pos2)
object_pos3 = np.array(object_pos3)
object_pos4 = np.array(object_pos4)

# Plot the graphical representations
fig = plt.figure(figsize=(10, 6))
fig_3 = plt.figure(figsize=(10, 6))

sub = fig.add_subplot(111)
sub1 = fig_3.add_subplot(111, projection='3d')

# Time plot of the postions of the vehicles and object
sub1.scatter(car_pos[:, 0], car_pos[:, 1], car_pos[:, 2], s=4,label='Car Position', color='purple')
sub1.scatter(object_pos1[:, 0], object_pos1[:, 1], object_pos1[:, 2], s=4,label='object_pos1', color='blue')
sub1.scatter(object_pos2[:, 0], object_pos2[:, 1], object_pos2[:, 2], s=4,label='object_pos2', color='red')
sub1.scatter(object_pos3[:, 0], object_pos3[:, 1], object_pos3[:, 2], s=4,label='object_pos3', color='green')
sub1.scatter(object_pos4[:, 0], object_pos4[:, 1], object_pos4[:, 2], s=4,label='object_pos4', color='orange')

# Define the 3D dimensions of the car's boundary
for x, y, z in car_pos:
    vertices = [[x - car_size, y - car_size, (z - car_size)],
                [x + car_size, y - car_size, (z - car_size)],
                [x + car_size, y + car_size, (z - car_size)],
                [x - car_size, y + car_size, (z - car_size)],
                [x - car_size, y - car_size, (z + car_size)],
                [x + car_size, y - car_size, (z + car_size)],
                [x + car_size, y + car_size, (z + car_size)],
                [x - car_size, y + car_size, (z + car_size)]
                ]
    
    faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],
             [vertices[7], vertices[6], vertices[5], vertices[4]],
             [vertices[0], vertices[4], vertices[5], vertices[1]],
             [vertices[3], vertices[2], vertices[6], vertices[7]],
             [vertices[0], vertices[3], vertices[7], vertices[4]],
             [vertices[1], vertices[5], vertices[6], vertices[2]]
             ]
    
    cube = Poly3DCollection(faces, edgecolor='black', linewidths=0.5, alpha=0.2)
    sub1.add_collection3d(cube)

sub1.set_xlabel('X Position (m)')
sub1.set_ylabel('Y Position (m)')
sub1.set_zlabel('Timestamp (s)')
sub1.set_title('Time Visualization of the Vechicle and Objects')
plt.grid(True)
plt.legend()

# Position plot of the object and vehicle irrespective of time
sub.scatter(df['FirstObjectDistance_X'], df['FirstObjectDistance_Y'], s=4,label='First Object', color='blue')
sub.scatter(df['SecondObjectDistance_X'], df['SecondObjectDistance_Y'],s=4, label='Second Object', color='red')
sub.scatter(df['ThirdObjectDistance_X'], df['ThirdObjectDistance_Y'], s=4,label='Third Object', color='green')
sub.scatter(df['FourthObjectDistance_X'], df['FourthObjectDistance_Y'], s=4,label='Fourth Object', color='orange')
sub.scatter(car_pos_x, car_pos_y, label='Car Position', s=4, color='purple')

# Plot the 2D boundary of the vehicle
for x, y in zip(*car_pos1):
    sub.add_patch(Rectangle(xy=(x-car_width/2, y-car_length/2), width=car_width, height=car_length,
                            linewidth=1, color='black', fill=False))
    
sub.set_xlabel('Position X(m)')
sub.set_ylabel('Position Y(m)')
sub.set_title('Position Plot')
sub.grid(True)
plt.legend()

fig_1 = plt.figure(figsize=(10, 6))
sub_1 = fig_1.add_subplot()
sub_1.plot(df['Timestamp'], car_acceleration_values, label='Acceleration (m/s^2)')
plt.xlabel('Time (S)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration Plot')
plt.grid(True)
plt.legend()
plt.show()