"""
==================================================================
Genetic Algorithm for Job Scheduling Optimization
==================================================================
Author: Aryan Baghinay  
Supervisor: Dr. Kolahan  
Ferdowsi University of Mashhad (FUM)

Description:
This program implements a **Genetic Algorithm (GA)** to optimize job scheduling problems.  
The objective is to minimize penalties due to late or early job completions by finding the optimal order of job execution.  

Features:
- User-friendly interface built with **Streamlit**.
- Flexible input options:
    1. Manual input of job data.
    2. Uploading an Excel file with job information.
- Customizable parameters for the Genetic Algorithm:
    - Population size
    - Selection rates (roulette, elitism, random transfer)
    - Crossover and mutation probabilities
    - Stopping criteria (number of iterations, CPU time, acceptable cost, or convergence).
- Visualization of results and insights into the scheduling process.

Key Functions:
- `initialize_population`: Creates the initial population of solutions.
- `calculate_fitness`: Evaluates the fitness of each solution based on penalties.
- `create_parent_pool`: Selects parents using elitism, roulette, and random selection.
- Crossover operations:
    - Order Crossover (OX)
    - Partially Mapped Crossover (PMX)
    - Cycle Crossover (CX)
- Mutation operations:
    - Swap mutation
    - Inversion mutation
    - Two-point swap mutation
- Supports multiple stopping criteria for flexible optimization.

How to Use:
1. Run the program using << streamlit run GA_APP.py >> in CMD (Command Prompt) at the directory where the file is saved.
2. Choose the input method (manual or Excel file).
3. Define the parameters for the Genetic Algorithm.
4. Start the optimization process and analyze the results.

Required Libraries:
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Time
- Random

"""






import streamlit as st 

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt




def get_input_with_default(label, default_value):
    user_input = st.text_input(label, value=', '.join(map(str, default_value)))
    if user_input.strip() == "":
        return default_value
    else:
        return [float(x.strip()) for x in user_input.split(',')]
    
def load_excel_file(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return None
    
    
def calculate_fitness(list):
    f=[]
    for gen in list:
        p=0
        c=np.zeros(20)
        for index ,job in enumerate(gen):
            if index==0:
                c[index]=t[job]
            else:
                c[index]=t[job]+c[index-1]
            p+=max(0, (c[index]-d[job])*alpha[job])+max(0, (d[job]-c[index])*beta[job])
            p=round(p, 3)
        f.append(p)
    return f



def C_calc(data):
        completion_time = np.zeros(N_J)
        for i, job in enumerate(data):
            if i == 0:
                completion_time[i] = t[job]
            else:
                completion_time[i] = completion_time[i-1] + t[job]
       
        return completion_time



def initialize_population(gen_size):
    population = []
    for i in range(gen_size):
        p = np.random.permutation(N_J).tolist()
        population.append(p)
    return population

def sort_gens(gen, fitness):
   
    combined = list(zip(fitness, gen))  
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=False)  

   
    fitness_scores_sorted, generations_sorted = zip(*combined_sorted)

  
    fitness_scores_sorted = list(fitness_scores_sorted)
    generations_sorted = list(generations_sorted)

    return generations_sorted, fitness_scores_sorted





def create_parent_pool(generations, fitness_scores, pool_size ):
    elite_count = int(elitism_rate * pool_size)  
    roulette_count = int(roulette_rate * pool_size)   
    random_count = pool_size - elite_count - roulette_count   

    
    elite_parents = generations[:elite_count]  
    
    
    total_fitness = sum(1/fitness for fitness in fitness_scores)  
    probabilities = [(1/fitness) / total_fitness for fitness in fitness_scores]  
    selected_parents_roulette = np.random.choice(len(generations), size=roulette_count, p=probabilities)
    roulette_parents = [generations[i] for i in selected_parents_roulette]
    
    
    random_parents = [generations[np.random.randint(len(generations))] for j in range(random_count)]
    
    
    parent_pool = elite_parents + roulette_parents + random_parents
    
    return parent_pool


 
def order_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(np.random.choice(size, size=2, replace=False))
    child = [-1] * size
    child[point1:point2] = parent1[point1:point2]
    current_index = point2
    for job in parent2:
        if job not in child:
            if current_index == size:
                current_index = 0
            child[current_index] = job
            current_index += 1
    return child



def mapped_crossover(parent1, parent2):
    size = len(parent1)
    
   
    a, b = sorted(random.sample(range(size), 2))
    
    
    child1 = parent1[:]
    child2 = parent2[:]
    
  
    child1[a:b+1] = parent2[a:b+1]
    child2[a:b+1] = parent1[a:b+1]
    
  
    for i in range(a, b+1):
        if child1[i] not in parent1[a:b+1]:
            for x in parent2:
                if x not in child1:
                    child1[i] = x
                    break
    
   
    for i in range(a, b+1):
        if child2[i] not in parent2[a:b+1]:
            for x in parent1:
                if x not in child2:
                    child2[i] = x
                    break
    
    return child1, child2



def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child1 = [-1] * size
    child2 = [-1] * size
    visited = [False] * size  

    cycle_start = 0
    while -1 in child1:   
        i = cycle_start
        cycle = []

       
        while not visited[i]:
            cycle.append(i)
            visited[i] = True   
            value = parent1[i]
            
            i = parent2.index(value) if value in parent2 else -1

        for i in cycle:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        cycle_start = (cycle_start + 1) % size

    return child1, child2






def random_crossover(parent1, parent2):
    crossover_type = np.random.choice(['OX','PMX'])  
    if crossover_type == 'OX':
        return (order_crossover(parent1, parent2), order_crossover(parent1, parent2))
    elif crossover_type == 'PMX':
        return mapped_crossover(parent1, parent2)
    elif crossover_type == 'CX':
        return cycle_crossover(parent1, parent2)
    
    
    

def swap_mutation(schedule):
    point1, point2 = np.random.choice(len(schedule), size=2, replace=False)
   
    schedule[point1], schedule[point2] = schedule[point2], schedule[point1]
    return schedule


def inversion_mutation(schedule):
    point1, point2 = np.random.choice(len(schedule), size=2, replace=False)
     
    if point1 > point2:
        point1, point2 = point2, point1   
    schedule[point1:point2+1] = schedule[point1:point2+1][::-1]
    return schedule


def two_point_swap_mutation(schedule):
    point1, point2 = np.random.choice(len(schedule), size=2, replace=False)
    schedule[point1], schedule[point2] = schedule[point2], schedule[point1]
    return schedule


def random_mutation(schedule):
    mutation_type = np.random.choice(['swap', 'inversion', 'two_point_swap'])  
    if mutation_type == 'swap':
        return swap_mutation(schedule)
    elif mutation_type == 'inversion':
        return inversion_mutation(schedule)
    elif mutation_type == 'two_point_swap':
        return two_point_swap_mutation(schedule)
 
    
    
    


jobs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
t = [9, 13, 26, 5, 6, 22, 15, 9, 7, 12, 8, 25, 12, 6, 11, 16, 19, 21, 17, 24]
d = [55, 220, 120, 5, 15, 25, 280, 185, 10, 20, 15, 40, 250, 30, 150, 80, 100, 40, 140, 100]
alpha = [0.9, 1.2, 0.5, 1.8, 1.4, 0.1, 1.5, 0.8, 1.9, 0.0, 1.9, 0.7, 1.0, 1.0, 1.7, 0.4, 0.2, 1.6, 1.9, 0.3]
beta = [0.1, 1.0, 0.1, 0.0, 0.4, 1.1, 0.2, 0.0, 0.1, 1.0, 0.2, 0.3, 0.2, 0.7, 0.5, 0.4, 1.2, 0.1, 0.5, 0.1]

cputime_max=9999999
convcheck=False
sfit=1
N_epoch=9999999
st.title("Genetic Algorithm for Job Scheduling")
st.write("This tool utilizes a genetic algorithm to optimize job scheduling.")
st.write("Developed by Aryan Baghiay, under the supervision of Dr. Kolahan.")
st.write("")
st.write("")
st.subheader("Data Input")
option = st.radio(
    "Select input method:",
    ("Manual Input", "Upload Excel File")
)

if option == "Manual Input":
    
    jobs = get_input_with_default("Enter the job numbers:", jobs)
    t = get_input_with_default("Enter the job duration):", t)
    d = get_input_with_default("Enter the Due dates:", d)
    alpha = get_input_with_default("Enter the alpha values:",alpha)
    beta = get_input_with_default("Enter the beta values:", beta)
    input_lengths = [len(jobs), len(t), len(d), len(alpha), len(beta)]
    if len(set(input_lengths)) != 1:
        st.error("Error: All input lists must have the same length!")
    data = {
        "Job ": jobs,
        "jt": t,
        "jd": d,
        "j_alpha": alpha,
        "j_beta": beta
    }
    df = pd.DataFrame(data)
    df=df.T
   
    st.write("Data:")
    st.dataframe(df)

        
elif option == "Upload Excel File":
    
    st.markdown("""
### About the Data Input:
 **Excel File Input**: If you choose to upload an Excel file, the file should have the following columns:
   - **Job**: Job identifiers (e.g. 1 , 2 , 3 ...).(numeric values**Make sure the numbers start from 1 and are in sequential order.**
   - **jt**: Job processing times (numeric values).
   - **jd**: Job deadlines (numeric values).
   - **j_alpha**: Alpha values (numeric values).
   - **j_beta**: Beta values (numeric values).

   The file should have at least these 5 columns, and the column names should match exactly as mentioned.
""")

  
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
      
       df = load_excel_file(uploaded_file)

       if df is not None:
          
           st.write("Data loaded from Excel:")
           st.dataframe(df)

         
           required_columns = ["Job", "jt", "jd", "j_alpha", "j_beta"]
           missing_columns = [col for col in required_columns if col not in df.columns]

           if missing_columns:
               st.error(f"Missing columns: {', '.join(missing_columns)}")
           else:
              
               jobs = df["Job"].tolist()
               t = df["jt"].tolist()
               d = df["jd"].tolist()
               alpha = df["j_alpha"].tolist()
               beta = df["j_beta"].tolist()

              
               df_transposed = df.T

              
               st.write("Data from Excel:")
               st.dataframe(df_transposed)
       else:
           st.error("Failed to load the Excel file.")
        
st.subheader("Genetic Algorithm Parameter")

col1, col2 = st.columns(2)
with col1:
    gen_size = st.number_input("Generation Size", min_value=1, value=200, step=1, help="Number of members in each generation")
with col2:
    pool_size = st.number_input("Pool Size", min_value=1, value=100, step=1, help="Number of parents selected in each generation")

st.write(f"Generation Size: {gen_size}")
st.write(f"Pool Size: {pool_size}")
st.write("")


st.subheader("Parent Selection Parameters")
col1, col2 = st.columns(2)
with col1:
    roulette_rate = st.slider("Roulette Rate", min_value=0.0, max_value=1.0, value=0.7, step=0.05, help="Probability of using roulette wheel selection")
    elitism_rate = st.slider("Elitism Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Probability of using elitism selection")
    st.write(f"Roulette Rate: {roulette_rate}")
    st.write(f"Elitism Rate: {elitism_rate}")


with col2:
    random_rate = st.slider("Random Transfer Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Probability of transferring individuals randomly")
    st.write(f"Random Transfer Rate: {random_rate}")




st.subheader("Generation Parameters")
col1, col2 = st.columns(2)

with col1:
    r_m = st.slider("Mutation Probability (r_m)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Probability of mutation in the population")
    r_d = st.slider("Direct Transfer Probability (r_d)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Probability of direct transfer without crossover")
    st.write(f"Mutation Probability: {r_m}")
    st.write(f"Direct Transfer Probability: {r_d}")


with col2:
    r_c = st.slider("Crossover Probability (r_c)", min_value=0.0, max_value=1.0, value=0.9, step=0.05, help="Probability of crossover between parents")
    n_e = st.slider("Number of Elites (n_e)", min_value=1, max_value=200, value=5, step=1, help="Number of elite individuals to retain in the next generation")
    st.write(f"Crossover Probability: {r_c}")
    st.write(f"Number of Elites: {n_e}")
    
    
st.subheader("Stopping Criteria Selection")

stop_criteria = st.radio(
    "Select the stopping criteria",
    options=["Number of Iterations", "CPU Processing Time", "Acceptable Cost", "Convergence Criteria"],
    index=0,  
    help="Choose the stopping criteria for the algorithm"
)

if stop_criteria == "Number of Iterations":
    N_epoch = st.number_input("Number of Iterations ", value=250, min_value=1, help="The desired number of iterations")
    st.write(f"Selected Number of Iterations: {N_epoch}")
    SC='epoch'

elif stop_criteria == "CPU Processing Time":
    
    cputime_max = st.number_input("CPU Processing Time ", value=10, min_value=1, help="The desired CPU processing time in seconds")
    st.write(f"Selected CPU Processing Time: {cputime_max} seconds")
    SC='time'
if stop_criteria == "Acceptable Cost":
    
    sfit = st.number_input("Acceptable Cost ", value=197.0, min_value=0.0, help="The acceptable cost threshold")
    st.write(f"Selected Acceptable Cost: {sfit}")
    SC='Fitness'
elif stop_criteria == "Convergence Criteria":
   
    n_check = st.number_input("Generations with No Change ", value=50, min_value=1, help="The number of generations without change for stopping")
    st.write(f"Selected No Change in Generations: {n_check} generations")
    SC='conv'


N_J=len(jobs)
start_time = time.time()
best_solutions = []
best_fitness=[]
epoch=0
epochh=0
cputime=0
fit=10000
sfit=200
convcheck=False




st.subheader("Start Algorithm")
if st.button('Start Genetic Algorithm'):
    st.write("Running Genetic Algorithm...")
    
    while epoch in range(N_epoch) and cputime < cputime_max and fit > sfit and convcheck==False:
        if epochh==0:
            gen=initialize_population(gen_size)
            fitness=calculate_fitness(gen)
            gen,fitness=sort_gens(gen, fitness)
            best_solution_epoch=gen[0]
            best_solutions.append(gen[0])
            best_fitness_epoch=fitness[0]
            best_fitness.append(fitness[0])
            epochh=1
        parents=create_parent_pool(gen, fitness, pool_size )
        fitness=calculate_fitness(parents)
        parents,fitness=sort_gens(parents, fitness)
        gen=[]
        gen.extend(parents[:n_e])
        while len(gen)< gen_size:
            P1 , P2 =random.sample(parents, 2)
            r= random.random()
            if r <= (r_c):
                child1, child2 = random_crossover(P1, P2)
                if np.random.rand() < r_m:
                    child1 = random_mutation(child1)
                if np.random.rand() < r_m:
                    child2 = random_mutation(child2)
            else:
                child1=P1
                child2=P2
            
            gen.extend([child1, child2])
        fitness=calculate_fitness(gen)
        gen,fitness=sort_gens(gen, fitness)
        best_solution_epoch=gen[0]
        best_solutions.append(gen[0])
        best_fitness_epoch=fitness[0]
        best_fitness.append(fitness[0])
        if SC=='epoch':
            epoch+=1
        elif SC=='time':
            end_time = time.time()
            cputime = end_time - start_time
        elif SC=='Fitness':
            fit=best_fitness_epoch
        elif SC=='conv':
            last_n_elements = best_fitness[-n_check:]
            convcheck=len(set(last_n_elements)) == 1
            
        
        
    end_time = time.time()
    time = end_time - start_time
    st.subheader("Results and Outputs")
    st.write(f"Execution Time: {time:.2f} seconds")
    st.markdown(f"<h2 style='text-align: center;'>Minimum Calculated Cost: {fitness[0]}</h2>", unsafe_allow_html=True)
    
    
    Optimal_Scheduling=[x + 1 for x in gen[0]]

    index_names = ['Job', 't', 'd', 'C', 'L', 'T','alpha', 'beta', 'cost']
    df = pd.DataFrame(0, index=index_names, columns= \
                      [f'Column_{i+1}'for i in range(N_J)],dtype=float)
    df.loc["Job"] = Optimal_Scheduling


    tt=[t[j-1] for j in Optimal_Scheduling]
    df.loc["t"] = tt
    dd=[d[j-1] for j in Optimal_Scheduling]
    df.loc["d"] = dd
    aa=[alpha[j-1] for j in Optimal_Scheduling]
    df.loc["alpha"] = aa
    bb=[beta[j-1] for j in Optimal_Scheduling]
    df.loc["beta"] = bb

    cc=C_calc(gen[0])
    df.loc["C"] = cc
    ll=cc-dd
    df.loc["L"] = ll
    ttt=[abs(max(0,k)) for k in ll]
    df.loc["T"] = ttt

    cost=[max(0,value)*aa[index]+abs(min(0,value))*bb[index] for index, value in enumerate(ll)]
    df.loc["cost"] = cost
    
    st.write("Best Scheduling:")
    st.dataframe(df)
    
    plt.plot(best_fitness)
    plt.xlabel('Epoch')  
    plt.ylabel('Cost')  
    plt.title('Cost Graph Over Epoch')  
    plt.grid(True, linestyle='--', alpha=0.7)   
    plt.tight_layout()  
    st.pyplot(plt)
    
    st.subheader("Create Report") 
    columns = [f"Column{i+1}" for i in range(20)]  
    df_best_solutions = pd.DataFrame(best_solutions, columns=columns)
    df_best_fitness = pd.DataFrame(best_fitness, columns=[f"Column{i+1}" for i in range(1)])
    file_path1 = "best_solutions.xlsx"
    file_path2 = "best_fitness.xlsx"
    file_path3 = "Best_Scheduling.xlsx"
    df_best_solutions.to_excel(file_path1, index=False, engine='xlsxwriter')
    df_best_fitness.to_excel(file_path2, index=False, engine='xlsxwriter')
    df.to_excel(file_path3, index=False, engine='xlsxwriter')
    st.success(f"Excel file best solutions    saved as {file_path1}.")
    st.success(f"Excel file best fitness      saved as {file_path2}.")
    st.success(f"Excel file Best Scheduling   saved as {file_path3}.")
            
        
        
        
        
        
        
        
        
        
        
        
        


















