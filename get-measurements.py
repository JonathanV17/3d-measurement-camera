"""
Author: Jonathan A. Valadez Saldaña
Organisation: Universidad de Monterrey
Contact: jonathan.valadezg@udem.edu
First created: 4 Marzo 2024
Last updated: 5 Marzo 2024

Ejecutar script en terminal: python get-measurements.py --cam_index 0 --Z 100  --cal_file calibration_Jonathan.json 
"""

import cv2
import numpy as np
import json

def undistort_image(distorted_image, calibration_file):
    """
    Corrects distortion in the input image.
    
    Parameters:
        distorted_image (numpy.ndarray): Input distorted image.
        calibration_file (str): Path to the calibration file.
        Z (float): Distance from the camera to the object to measure.
        
    Returns:
        numpy.ndarray: Undistorted image.
    """
    # Cargar los parámetros de calibración desde el archivo JSON
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    
    # Extraer los parámetros de calibración necesarios
    camera_matrix = np.array(calibration_data['camera_matrix'])
    distortion_coefficients = np.array(calibration_data['distortion_coefficients'])
    
    # Corregir la distorsión en la imagen
    undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coefficients)
    
    return undistorted_image

def compute_line_segments(points, image, Z):
    """
    Computes the length of each line segment between consecutive points.
    
    Parameters:
        points (list of tuples): List of (x, y) coordinates of selected points.
        image (numpy.ndarray): Image where the line segments are drawn.
        Z (float): Distance from the camera to the object to measure.
        
    Returns:
        list of float: List of lengths of line segments.
    """
    line_lengths = []

    for i in range(len(points) - 1):
        # Calcular la distancia entre dos puntos consecutivos
        point1 = points[i]
        point2 = points[i+1]
        distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

        # Escalar la distancia en función de Z
        scaled_distance = distance * Z / 100  # Suponiendo que Z está en centímetros y necesitas convertirlo a metros

        line_lengths.append(scaled_distance)
        
        # Dibujar una línea entre los puntos en la imagen
        cv2.line(image, point1, point2, (255, 0, 255), 2)  # Dibujar una línea morada

    # Ordenar las distancias en orden ascendente
    line_lengths.sort()
    
    # Imprimir las distancias en orden ascendente
    print("Line lengths in ascending order:")
    for length in line_lengths:
        print(length)
    
    # Mostrar la imagen con las líneas dibujadas
    cv2.imshow('Line Segments', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return line_lengths


def compute_perimeter(line_lengths):
    """
    Computes the total perimeter from line segment lengths.
    
    Parameters:
        line_lengths (list of float): List of lengths of line segments.
        
    Returns:
        float: Total perimeter.
    """
    if not line_lengths:
        print("No points selected. Perimeter cannot be calculated.")
        return 0
    
    total_perimeter = sum(line_lengths)
    print("Perimeter:", total_perimeter)
    return total_perimeter

def main(cam_index, Z, calibration_file):
    """
    Main function to interact with the user and perform measurements.
    
    Parameters:
        cam_index (int): Index of the camera.
        Z (float): Distance from the camera to the object to measure.
        calibration_file (str): Path to the calibration file.
    """
    # Inicialización de la cámara
    cap = cv2.VideoCapture(cam_index)

    # Variables para almacenar los puntos seleccionados por el usuario
    selected_points = []

    # Función para manejar eventos del mouse
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            print("Point selected:", (x, y))
            # Dibujar un círculo en el punto seleccionado
            cv2.circle(undistorted_frame, (x, y), 5, (0, 255, 0), -1)  # Dibujar un círculo verde
            cv2.imshow('Undistorted Image', undistorted_frame)

        elif event == cv2.EVENT_MBUTTONDOWN:
            print("Selection stopped.")
            if len(selected_points) >= 2:
                selected_points.append(selected_points[0])
                # Calcular las longitudes de los segmentos de línea y dibujar las líneas
                compute_line_segments(selected_points, undistorted_frame, Z)
                # Limpiar la lista de puntos seleccionados para una nueva selección
                selected_points.clear()
                return  # Salir del bucle cuando se detenga la selección de puntos
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # Manejar clic derecho
            print("Deleting all selected points.")
            selected_points.clear()  # Eliminar todos los puntos seleccionados
            # Redibujar la imagen con los puntos actualizados (en este caso, sin puntos)
            cv2.imshow('Undistorted Image', undistorted_frame)
            
    # Establecer el manejador de eventos del mouse
    cv2.namedWindow('Undistorted Image')
    cv2.setMouseCallback('Undistorted Image', mouse_callback)
    
    # Bucle principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Corrección de distorsión en la imagen
        undistorted_frame = undistort_image(frame, calibration_file)
        
        # Mostrar la imagen corregida
        cv2.imshow('Undistorted Image', undistorted_frame)
        
        # Esperar a que el usuario presione una tecla para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la captura y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vision-based measurements script')
    parser.add_argument('--cam_index', type=int, default=0, help='Index of the camera')
    parser.add_argument('--Z', type=float, required=True, help='Distance from camera to object (in cm)')
    parser.add_argument('--cal_file', type=str, required=True, help='Path to calibration file')
    args = parser.parse_args()
    
    main(args.cam_index, args.Z, args.cal_file)
