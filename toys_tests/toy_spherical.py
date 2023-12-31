import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def generate_spherical_data(num_points, radius=1.0, data_class=1):
    """
    Generate random points on the surface of a sphere.

    Parameters:
    - num_points: The number of points to generate.
    - radius: The radius of the sphere.
    - data_class: class of objects

    Returns:
    - An array of shape (num_points, 3) containing (x, y, z) coordinates of the points on the sphere.
    """
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    sin_theta = np.sqrt(1 - cos_theta**2)
    class_vec = np.full(num_points, data_class)

    x = radius * sin_theta * np.cos(phi)
    y = radius * sin_theta * np.sin(phi)
    z = radius * cos_theta

    spherical_data = np.column_stack((x, y, z))
    spherical_data_class = np.column_stack((x, y, z, class_vec))
    return spherical_data, spherical_data_class


def generate_cylindrical_data(num_points, radius=1.0, data_class=1):
    """
    Generate random points on the surface of a cylinder.

    Parameters:
    - num_points: The number of points to generate.
    - radius: The radius of the cylinder.
    - data_class: class of objects

    Returns:
    - An array of shape (num_points, 3) containing (x, y, z) coordinates of the points on the sphere.
    """
    radius = np.random.uniform(radius-1, radius, num_points)  # Random radius values
    theta = np.random.uniform(0, 2*np.pi, num_points)  # Random angle values
    height = np.random.uniform(1, 10, num_points)  # Random height values
    class_vec = np.full(num_points, data_class)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = height

    cylindrical_data = np.column_stack((x, y, z))
    cylindrical_data_class = np.column_stack((x, y, z, class_vec))
    return cylindrical_data, cylindrical_data_class


# Generate spherical data
num_points = 1000
radius1 = 5.0
radius2 = 10.0
spherical_data_1, spherical_data_1_class = generate_spherical_data(num_points, radius1, 1)
spherical_data_2, spherical_data_2_class = generate_spherical_data(num_points, radius2, -1)

print(spherical_data_1_class)

df_sph1 = pd.DataFrame(spherical_data_1_class, columns = ['X','Y','Z', 'class'])
df_sph2 = pd.DataFrame(spherical_data_2_class, columns = ['X','Y','Z', 'class'])
df_tot = pd.concat([df_sph1, df_sph2], axis=0)
#print(df_tot)
df_tot.to_csv('spherical_data.csv', index=False)

# Plot spherical data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(spherical_data_1[:, 0], spherical_data_1[:, 1], spherical_data_1[:, 2], c='b', marker='o')
ax.scatter(spherical_data_2[:, 0], spherical_data_2[:, 1], spherical_data_2[:, 2], c='r', marker='x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Spherical Data Plot')
plt.show()
plt.clf()

# genenerate cylindrical data
num_points = 1000
radius1 = 5.0
radius2 = 10.0
cylindrical_data_1, cylindrical_data_1_class = generate_cylindrical_data(num_points, radius1, 1)
cylindrical_data_2, cylindrical_data_2_class = generate_cylindrical_data(num_points, radius2, -1)

df_cyl1 = pd.DataFrame(cylindrical_data_1_class, columns = ['X','Y','Z', 'class'])
df_cyl2 = pd.DataFrame(cylindrical_data_2_class, columns = ['X','Y','Z', 'class'])
df_tot_c = pd.concat([df_cyl1, df_cyl2], axis=0)
df_tot.to_csv('cylindrical_data.csv', index=False)

# store sum
df_sum_sc_1 = pd.DataFrame(cylindrical_data_1_class+spherical_data_1_class, columns = ['X','Y','Z', 'class'])
df_sum_sc_2 = pd.DataFrame(cylindrical_data_2_class+spherical_data_2_class, columns = ['X','Y','Z', 'class'])
df_tot_sum_c = pd.concat([df_cyl1, df_cyl2], axis=0)
df_tot.to_csv('cylindrical_spherical_data.csv', index=False)

# Plot cylindrical data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cylindrical_data_1[:, 0], cylindrical_data_1[:, 1], cylindrical_data_1[:, 2], c='b', marker='o')
ax.scatter(cylindrical_data_2[:, 0], cylindrical_data_2[:, 1], cylindrical_data_2[:, 2], c='r', marker='x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cylindrical Data Plot')
plt.show()
plt.clf()


# Plot sum of data
#df_add = df1.add(df2, fill_value=0)
#df_cyl1
#df_cyl2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter((cylindrical_data_1+spherical_data_1)[:, 0], (cylindrical_data_1+spherical_data_1)[:, 1], (cylindrical_data_1+spherical_data_1)[:, 2], c='b', marker='o')
ax.scatter((cylindrical_data_2+spherical_data_2)[:, 0], (cylindrical_data_2+spherical_data_2)[:, 1], (cylindrical_data_2+spherical_data_2)[:, 2], c='r', marker='x')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cylindrical Data Plot')
plt.show()
plt.clf()