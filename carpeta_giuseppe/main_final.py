import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from sympy import symbols, sympify
import cvxpy as cp
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


class MAIN(QMainWindow):
    def __init__(self,parent = None):
        super().__init__(parent)
        uic.loadUi("VERSION1.ui", self)

        self.pushButton_5.clicked.connect(self.activar_programacion_convexa)
        self.pushButton_6.clicked.connect(self.activar_programacion_no_convexa)
        self.pushButton_7.clicked.connect(self.activar_programacion_geometrica)
        self.pushButton_8.clicked.connect(self.activar_programacion_fraccional)
        self.pushButton_9.clicked.connect(self.activar_problema_complementariedad)
    def activar_programacion_convexa(self):
        self.ocultar_menu()
        self.programacion_convexa = P_C(self)
        self.programacion_convexa.show()

    def activar_programacion_no_convexa(self):
        self.ocultar_menu()
        self.programacion_no_convexa = P_N_C(self)
        self.programacion_no_convexa.show()

    def activar_programacion_geometrica(self):
        self.ocultar_menu()
        self.programacion_geometrica = P_G(self)
        self.programacion_geometrica.show()

    def activar_programacion_fraccional(self):
        self.ocultar_menu()
        self.programacion_fraccional = P_F(self)
        self.programacion_fraccional.show()

    def activar_problema_complementariedad(self):
        self.ocultar_menu()
        self.problema_complementariedad = P_CO(self)
        self.problema_complementariedad.show()


    def ocultar_menu(self):
        self.hide()

class P_C(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        uic.loadUi("UI_PROGRAMCION_CONVEXA.ui", self)
        self.main_window = main_window
        self.pushButton_2_c.clicked.connect(self.regresar_main)
        self.pushButton_c.clicked.connect(self.resolver_optimizacion_convexa)

    def resolver_optimizacion_convexa(self):
        try:
            # Obtener la función objetivo y las restricciones desde los campos de texto
            func_obj_text = self.ingresar_convexa.text()
            restricciones_text = self.textEdit_c.toPlainText()

            # Convertir las cadenas de texto en expresiones de CVXPY
            x = cp.Variable()
            y = cp.Variable()

            func_obj = eval(func_obj_text, {'x': x, 'y': y})
            
            # Dividir las restricciones y evaluarlas individualmente
            restricciones = [eval(restriccion.strip(), {'x': x, 'y': y}) for restriccion in restricciones_text.split(';') if restriccion]

            # Definir el problema de optimización
            problema = cp.Problem(cp.Minimize(func_obj), restricciones)

            is_convex = problema.is_dcp()

            if is_convex:
                problema.solve()

            # self.plot_graph(func_obj, x, y, x.value, y.value, "Función Objetivo Convexa")


            # Mostrar los resultados en la interfaz
                self.label_4_c.setText(f"VALOR OPTIMO DE X: {x.value}")
                self.label_5_c.setText(f"VALOR OPTIMO DE Y: {y.value}")
                self.label_6_c.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: {func_obj.value}")
            else:
                self.label_4_c.setText(f"VALOR OPTIMO DE X: NO ES CONVEXO")
                self.label_5_c.setText(f"VALOR OPTIMO DE Y: NO ES CONVEXO")
                self.label_6_c.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: NO ES CONVEXO")
            # Resolver el problema
            
        except Exception as e:
            self.label_6_c.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: {e}") 

    #def plot_graph(self, func_obj, x, y, x_opt, y_opt, title):
        # Crea un rango de valores para x e evalúa la función objetivo
     #   x_vals = [val for val in range(-10, 11)]
      #  y_vals = [func_obj.subs({x: val_x, y: val_x}) for val_x in x_vals]

        # Grafica la función objetivo
       # plt.plot(x_vals, y_vals, label='Función Objetivo')

        # Resalta el punto óptimo
        #plt.scatter([x_opt], [y_opt], color='red', label='Punto Óptimo')

        #plt.title(title)
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.legend()
        #plt.show()

    def regresar_main(self):
        self.close()  # Cierra la ventana actual
        self.main_window.show()  # Muestra la ventana principal

class P_N_C(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        uic.loadUi("UI_PROGRMACION_NO_CONVEXA.ui", self)

        self.main_window = main_window
        self.pushButton_2_n_c.clicked.connect(self.regresar_main)
        self.pushButton_n_c.clicked.connect(self.resolver_optimizacion_no_convexa)

    def resolver_optimizacion_no_convexa(self):
        function_text = self.ingresa_no_convexa.text()

        try:
            # Intenta evaluar la función ingresada
            objective_function = lambda x: eval(function_text)
            x_min, x_max = -5, 5

            x_opt, f_opt = self.find_global_optimum(objective_function, x_min, x_max)

            result_text = f'El óptimo global encontrado es: x = {x_opt:.4f} con f(x) = {f_opt:.4f}'
        except Exception as e:
            result_text = f'Error al evaluar la función: {str(e)}'

        self.label_4.setText(result_text)

    def find_global_optimum(self, objective_function, x_min, x_max, num_iterations=100000):
        x_opt = None
        f_opt = float('inf')

        for i in range(num_iterations):
            x = random.uniform(x_min, x_max)
            f = objective_function(x)

            if f < f_opt:
                x_opt = x
                f_opt = f

        return x_opt, f_opt



    def regresar_main(self):
        self.close()  # Cierra la ventana actual
        self.main_window.show()  # Muestra la ventana principal


class P_G(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        uic.loadUi("UI_PROGRAMACION_GEOMETRICA.ui", self)

        self.main_window = main_window
        self.pushButton_2_g.clicked.connect(self.regresar_main)
        self.pushButton_g.clicked.connect(self.resolver_optimizacion)

    def resolver_optimizacion(self):
        try:
            # Obtener la función objetivo y las restricciones desde los campos de texto
            func_obj_text = self.ingresar_geometrica.text()
            restricciones_text = self.textEdit_g.toPlainText()

            # Convertir las cadenas de texto en expresiones de funciones
            def objective_function(xy):
                x, y = xy
                return eval(func_obj_text, {'x': x, 'y': y})

            def constraint_function(xy):
                return [eval(restriccion.strip(), {'x': xy[0], 'y': xy[1]}) for restriccion in restricciones_text.split(';') if restriccion]


            # Punto de inicio
            initial_guess = [1.0, 2.0]

            # Resolver el problema de optimización
            result = minimize(objective_function, initial_guess, constraints={'type': 'ineq', 'fun': constraint_function})

            # Mostrar resultados
            self.label_4_g.setText(f"VALOR OPTIMO DE X: {result.x}")
            self.label_5_g.setText(f"VALOR OPTIMO DE Y: {result.x}")
            self.label_6_g.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: {result.fun}")

        except Exception as e:
            print(f"Error: {e}")

    def regresar_main(self):
        self.close()  # Cierra la ventana actual
        self.main_window.show()  # Muestra la ventana principal

class P_F(QMainWindow):
    
    def __init__(self,main_window):
        super().__init__()
        uic.loadUi("UI_PROGRAMACION_FRACCIONAL.ui", self)
        self.main_window = main_window
        self.pushButton_2_f.clicked.connect(self.regresar_main)
        self.pushButton_f.clicked.connect(self.resolver_optimizacion_fraccional)

    def resolver_optimizacion_fraccional(self):
        try:
            # Obtener la función objetivo y las restricciones desde los campos de texto
            func_obj_text = self.ingresar_fraccional.text()
            restricciones_text = self.textEdit_f.toPlainText()

            # Convertir las cadenas de texto en expresiones de CVXPY
            x = cp.Variable()
            y = cp.Variable()

            func_obj = eval(func_obj_text, {'x': x, 'y': y})
            
            # Dividir las restricciones y evaluarlas individualmente
            restricciones = [eval(restriccion.strip(), {'x': x, 'y': y}) for restriccion in restricciones_text.split(';') if restriccion]

            # Definir el problema de optimización
            problema = cp.Problem(cp.Minimize(func_obj), restricciones)

            is_convex = problema.is_dcp()

            if is_convex:
                problema.solve()

            # self.plot_graph(func_obj, x, y, x.value, y.value, "Función Objetivo Convexa")


            # Mostrar los resultados en la interfaz
                self.label_4_f.setText(f"VALOR OPTIMO DE X: {x.value}")
                self.label_5_f.setText(f"VALOR OPTIMO DE Y: {y.value}")
                self.label_6_f.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: {func_obj.value}")
            else:
                self.label_4_f.setText(f"VALOR OPTIMO DE X: NO ES CONVEXO")
                self.label_5_f.setText(f"VALOR OPTIMO DE Y: NO ES CONVEXO")
                self.label_6_f.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: NO ES CONVEXO")
            # Resolver el problema
            
        except Exception as e:
            self.label_6_f.setText(f"VALOR OPTIMO DE LA FUNCION OBJETIVO: {e}") 

    def regresar_main(self):
        self.close()  # Cierra la ventana actual
        self.main_window.show()  # Muestra la ventana principal

class P_CO(QMainWindow):
    def __init__(self,main_window):
        super().__init__()
        uic.loadUi("PROBLEMA_COMPLEMENTARIEDAD.ui",self)
        self.main_window = main_window
        self.pushButton_com.clicked.connect(self.actualizar_tabla)
        self.pushButton_com_resolver.clicked.connect(self.calcular_P_CO)
        self.pushButton_2_f.clicked.connect(self.regresar_main)
    def actualizar_tabla(self):
        try:
            columanas_filas = int(self.ingresar_c_f.text())
            self.tableWidget.setRowCount(columanas_filas)
            self.tableWidget.setColumnCount(columanas_filas)

            self.tableWidget_2.setRowCount(columanas_filas)
            self.tableWidget_2.setColumnCount(1)

            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()

            self.tableWidget_2.resizeColumnsToContents()
            self.tableWidget_2.resizeRowsToContents()
        except Exception as e:
            
            self.label_5_co.setText(f"ALGUN ERROR: {e}")


    def calcular_P_CO(self):
        try:
            o_matriz = self.tableWidget.rowCount()
            Matriz = np.zeros((o_matriz, o_matriz))
            # obtenemos datos de la matriz
            for i in range(o_matriz):
                for j in range(o_matriz):
                    valor = self.tableWidget.item(i,j)
                    if valor is not None:
                        Matriz[i,j] = float(valor.text())

            #obtenemos los datos del vector
            vector = self.tableWidget_2.rowCount()
            q = np.zeros(vector)

            for i in range(vector):
                item = self.tableWidget_2.item(i,0)
                if item is not None:
                    q[i] = float(item.text())

            #definir la funcion objetivo
            def funcion_objetivo(x):
                return np.dot(x[:-1], Matriz.dot(x[:-1])+q)
            
            constrints = [{'type': 'eq', 'fun': lambda x, i=i: Matriz[i, :].dot(x[:-1]) - q[i]} for i in range(vector)]

            # Especificar punto de inicio
            initial_guess = np.ones(vector + 1)

            # Resolver el problema de optimización
            result = minimize(funcion_objetivo, initial_guess, constraints=constrints)
        
            # Mostrar resultados
            self.label_6_co.setText(f"Resultado de la optimización:\nx = {result.x[:-1]}\nValor de la función objetivo: {result.fun}")
            
        except ValueError as e:
            self.label_6_co.setText(f"Error: {str(e)} - Ingrese valores numéricos para filas y columnas.")
        except Exception as e:
            self.label_6_co.setText(f"Error: {str(e)}")

    def regresar_main(self):
        self.close()  # Cierra la ventana actual
        self.main_window.show()  # Muestra la ventana principal

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = MAIN()
    GUI.show()
    sys.exit(app.exec_())