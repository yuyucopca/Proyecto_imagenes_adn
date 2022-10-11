# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import pdb 
# from numpy import argmax
# from numpy import array
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

Direccion = 'C:/Users/yulen/OneDrive/Documentos/CIDESI/Maestria/Proyecto de tesis/Codigo/Secuencias_Breast_mutations/SeqBRCA1_Benignas/'
nombre_arch = input("Colocar nombre del archivo con extension: ")

Sec_01 = open(Direccion + nombre_arch,'r')
Sec_01.readline()
b=Sec_01.read()
#print('b ', b)

b3 = list(b)
b3 = list(filter(('(').__ne__, b3))
b3 = list(filter((')').__ne__, b3))
b3 = list(filter((' ').__ne__, b3))
b3 = list(filter(('0').__ne__, b3))
b3 = list(filter(('1').__ne__, b3))
b3 = list(filter(('2').__ne__, b3))
b3 = list(filter(('3').__ne__, b3))
b3 = list(filter(('4').__ne__, b3))
b3 = list(filter(('5').__ne__, b3))
b3 = list(filter(('6').__ne__, b3))
b3 = list(filter(('7').__ne__, b3))
b3 = list(filter(('8').__ne__, b3))
b3 = list(filter(('9').__ne__, b3))
b3 = list(filter(('\n').__ne__, b3))
b3 = list(filter(('-').__ne__, b3))
#print(b3)
#tamano_lista = len(b3)
#print(tamano_lista)
init_pos = int(input("Colocar posicion de inicio de la cadena: "))

def snv():
    change_pos = int(input("Colocar la posicion donde hay cambio: "))
    local_change_pos = change_pos - init_pos 
    #print(local_change_pos)
    b3[local_change_pos] = input("Colocar snv: ")
    #print(b3) 
    nombre_arch_nuevo = input("Colocar nombre de ID: ")
    Descripcion_primera_linea = input("Colocar la descripcion del archivo: ")
    new_txt_arch = open(Direccion + nombre_arch_nuevo, 'w')
    new_txt_arch.write(Descripcion_primera_linea)
    new_txt_arch.write("\n"+str(b3))
    new_txt_arch.close()

def deletion():
    deletions_pos_init = int(input("Colocar la posicion de delecion inicial: "))
    deletions_pos_final = int(input("Colocar la posicion de delecion final: ")) + 1
    elementoABorrar = deletions_pos_init - init_pos
    del b3[elementoABorrar:deletions_pos_final]
    #print(b3)
    nombre_arch_nuevo = input("Colocar nombre de ID: ")
    Descripcion_primera_linea = input("Colocar la descripcion del archivo: ")
    new_txt_arch = open(Direccion + nombre_arch_nuevo, 'w')
    new_txt_arch.write(Descripcion_primera_linea)
    new_txt_arch.write("\n"+str(b3))
    new_txt_arch.close()

print("snv o deletion? ")
opcion = input()

while(opcion != 'snv' and opcion != 'delecion'):
    print("Lo siento, no te entiendo... :'( \n Vuelvelo a intentar porfa:")
    print("¿Deseas probar una opcion? snv/delecion")
    opcion = input()


if(opcion == 'snv'):
    print("Acabas de elegir snv")
    snv()
elif(opcion == 'delecion'):
    print("Acabas de eligir Delecion:")
    deletion()  