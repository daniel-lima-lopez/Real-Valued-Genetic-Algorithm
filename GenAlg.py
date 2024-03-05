import numpy as np

class RealGA:
    def __init__(self, nv, f, ni=10, ri = -10.0, rf=10.0, pc=0.9, pm=0.05, el=1):
        self.nv = nv # numero de variables en el genotipo
        self.ri = ri # rango inicial de las variables
        self.rf = rf # rango final de las variables
        self.ni = ni # numero de individuos en la poblacion
        self.f = f # funcion a minimizar

        self.pc = pc # probabilidad de crossover
        self.pm = pm # probabilidad de mutacion
        self.el = el # numero de padres a conservar en el algoritmo

        # creacion de la poblacion inicial y lista de padres e hijos
        self.population = np.random.uniform(self.ri, self.rf, (self.ni, self.nv))
        self.parents = []
        self.childs = []
    
    def torneo_bin(self):
        fs = self.f(self.population) # evaluacion de f

        # jugamos ni torneos binarios eligiendo aleatoriamente 2 contrincantes
        pool = []
        for i in range(self.ni):
            r1 = np.random.randint(0,self.ni) # primer contrincante
            r2 = np.random.randint(0,self.ni) # segundo contrincante
            rs = [r1, r2] # indices de los individuos seleccionados
            gs = np.array([fs[ri] for ri in rs]) # fitnes de los contrincantes

            # agregamos el de fit menor
            mi = np.argsort(gs)[0]
            pool.append(self.population[rs[mi]])
        
        # al finalizar declaramos como padres a los ganadores de los torneos binarios
        self.parents = np.array(pool)
    
    def crossover(self):
        # creamos hijos con los padres seleccionados en la etapa anterior
        self.childs = []
        for i in range(0,self.ni,2): # las parejas se forman con pares de individuos continguos
            # el crossover se realiza si se cumple con una probabilidad pc
            if np.random.choice([True, False], p=[self.pc, 1-self.pc]): # verificamos si se debe hacer crossover
                aux_p = np.random.randint(0, self.nv) # seleccionamos un pivote para mezclar variables
                c1 = self.parents[i].copy()
                c2 = self.parents[i+1].copy()

                # mezclamos las variable promediando despues del pivote
                c1[aux_p:] = 0.5*self.parents[i][aux_p:] + 0.5*self.parents[i+1][aux_p:]
                c2[aux_p:] = 0.5*self.parents[i][aux_p:] + 0.5*self.parents[i+1][aux_p:]
                self.childs += [c1, c2]
            
            # si no se cumple la probabilidad
            else:
                self.childs += [self.parents[i].copy(), self.parents[i+1].copy()]
        self.childs = np.array(self.childs)
    
    def mutation(self):
        for ind in range(self.ni): # iteramos sobre los individuos
            for vi in range(self.nv): # iteramos sobre las variables 
                # si cumple la probabilidad de mutacion pm
                if np.random.choice(a=[False,True], p=[1-self.pm, self.pm]):
                    self.childs[ind][vi] = np.random.uniform(self.ri, self.rf) # elige un nuevo valor dentro del rango

    def reemplazo_gen(self):
        # seleccionamos los mejores padres
        fs = np.array(self.f(self.parents)) # evalua el fitnes de los individuos
        kis = np.argsort(fs)
        aux_pool1 = [self.parents[kis[i]] for i in range(self.el)]

        # seleccionamos los mejores hijos
        fs = np.array(self.f(self.childs)) # evalua el fitnes de los individuos
        kis = np.argsort(fs)
        aux_pool2 = [self.childs[kis[i]] for i in range(self.ni-self.el)]

        # combinamos
        pool = aux_pool1 + aux_pool2
        self.population = np.array(pool)

    def train_estb(self, gens, pob_info=False):
        bests = [] #mejores individuos de cada generacion
        best_fs = [] # evaluacion de los mejores individuos        
        
        # poblacion incial
        fs = self.f(self.population)
        min_i = np.argmin(fs)
        min_f = fs[min_i]

        print(f'START')
        print(f'Fit promedio : {np.sum(fs)/len(fs)}')
        print(f'Fit minimo ({self.population[min_i]}): {min_f}')
        print(f'Poblacion:\n{self.population}')
        print(f'fs:\n{fs}')
        bests.append(self.population[min_i])
        best_fs.append(min_f)

        for gi in range(gens):
            self.torneo_bin()
            self.crossover()
            self.mutation()
            self.reemplazo_gen() # se cambian los padres por los hijos

            fs = self.f(self.population)
            min_i = np.argmin(fs)
            min_f = fs[min_i]
            print(f'\nGeneracion: {gi+1}')
            print(f'Fit promedio : {np.sum(fs)/len(fs)}')
            print(f'Fit minimo ({self.population[min_i]}): {min_f}')
            if pob_info:
                print(f'Poblacion:\n{self.population}')
                print(f'fs:{fs}')
            bests.append(self.population[min_i])
            best_fs.append(min_f)
        
        return np.array(bests), np.array(best_fs)