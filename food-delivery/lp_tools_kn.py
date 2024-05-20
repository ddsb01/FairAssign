import numpy as np

def L2Distance(data):
  """
  returns a [num_dirvers x num_drivers] shaped symmetric matrix giving distance between all drivers 
  """
  # "data": latitude-longitude level locations 
  transposed = np.expand_dims(data, axis = 1)
  distance = np.power(data - transposed, 2)
  distance = np.power(np.abs(distance).sum(axis = 2), 0.5) 
  # distance = distance*110 # why multiply by 110? Is it Euclidean to Haversine??

  return distance 

def abs_difference(ratings):
    transposed = np.expand_dims(ratings, axis=1)
    diff = abs(ratings-transposed) 
    return diff 

def minmax(distance, fair_distance):
    num_samples = len(distance)
    # Normalize 'distance': #cwp
    mx, mn = distance.max(), distance.min()
    dists = distance.flatten()
    dists = np.asarray( [((x-mn)/(mx-mn)) for x in dists] )
    distance = dists.reshape((num_samples, num_samples))

    # Also scale the fair_distance:
    fair_distance = (fair_distance-mn)/(mx-mn)
    
    return distance, fair_distance


def cost_function(dataset, centres, num_variables, prohibited_assignments):
  """
  Given the dataset and cluster centres, this function returns the distance between all samples and cluster centres and returns the lp objective coefficients
  Input:
    dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
    centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
  Output:
    objective - [ num_variables ] 1D numpy matrix containing weights for corresponding variables
  """
  transposed = np.expand_dims(dataset, axis = 1)  
  distance = np.power(transposed - centres, 2).sum(axis = 2)
  distance = distance.reshape(-1)		# Row major reshaping

  zero_vector = np.zeros(num_variables - len(distance))	# Pad the rest with 0s. They don't contribute to cost
  objective = np.concatenate([distance, zero_vector], axis = 0)

  # # Normalize and scale the 'objective':  
  # # Normalizing this way makes no difference to the final solution but may help in numerical stability.
  # # Also, note that normalizing for each row separately is not really required bcz \sigma x_vc = 1 takes care of scale mismatch of distances (from centres) of individual drivers.
  # # this scaling may affect numerical stability of the problem
  mx, mn = objective.max(), objective.min() 
  objective = np.asarray( [((x-mn)/(mx-mn)) for x in objective] )
  # Scaling the entire objective matrix, rather than each row separately, is the correct approach since each value represents physical distance! And we want to minimize the global \sigma {physical distance} for cost-efficiency

  ############################################################################################################################
  # making the objective values corresponding to a variable P_i_k higher lead to lesser value of P_i_k in the solution
  # making objective[P_i_k]->inf is equivalent to implementing a restricted assignment where driver "i" can never be assigned to ffc "k"
  prohibited_assignments = prohibited_assignments.flatten()
  
  INF = 1e3 # since all other objective values are between [0, 1]
  for idx in range(len(prohibited_assignments)):
    if prohibited_assignments[idx]:
        objective[idx] = INF
        
  # This change will cause the dual objective values to be high, don't worry about that
  ############################################################################################################################

  return objective




def prepare_to_add_variables(dataset, centres, prohibited_assignments):

  """
  Assumes TV distance as statistical distance metric. 
  Takes as input the dataset and cluster centres proposed by some clustering algorirthm and returns the weight vectors
  Input:
    dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
    centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
    prohibited_assignments - [ num_samples x num_clusters ] 2D numpy matrix set for the assignments which are to be disallowed by setting high 'objective' for them.
  Output:
    objective - [ num_variables ] 1D numpy array containing the weight of different variables
    lower_bound - [ num_variables ] 1D array specifying the minimum value of a variable
    upper_bound - [ num_variables ] 1D array specifying the maximum value of a variable
    variable_names - [ num_variables ] 1D sting list consisting of the names of variables 
    P, C - P and C array contain the id of the corresponding variables
  Details:
    P_i_k is a real probability variable that denotes the weight for connecting ith sample to kth cluster
    ""C_i_j_k is a intermediate variable that is an upper bound on | P_i_k - P_j_k | - A hack to convert non linear abs constraint to a linear constraint""
    
    
    P_i_k => probability that driver 'i' will be assigned to ffc 'k'
  """

  num_samples = len(dataset)
  num_centres = len(centres)
  _id = 0

  # P and C array contain the id of the corresponding variables
  P = np.zeros([num_samples, num_centres]).astype(int)
  C = np.zeros([num_samples, num_samples, num_centres]).astype(int)

  probability_variables = []
  for _point in range(num_samples):
    for _centre in range(num_centres):

      probability_variables.append("P_{point}_{centre}".format(
          point = _point,
          centre = _centre
        ))
      # Keep track of P_i_j's position in the lp variable vector 
      P[_point][_centre] = _id
      _id += 1
 
  abs_constraint_variables = []
  for _point1 in range(num_samples):
    for _point2 in range(_point1 + 1, num_samples):
      for _centre in range(num_centres):

        abs_constraint_variables.append("C_{point1}_{point2}_{centre}".format(
            point1 = _point1,
            point2 = _point2,
            centre = _centre
          ))
        # Keep track of C_i_j_k's position in lp variable vector
        C[_point1][_point2][_centre] = C[_point2][_point1][_centre] = _id
        _id += 1


  # Concatenating the names of both the types of variables
  variable_names = probability_variables + abs_constraint_variables


  # Setting lower bound = 0 and upper bound = 1 for all the variables: 0 <= x_{ij} <= 1 constraint
  num_variables = len(variable_names)
  lower_bound = [0.0 for i in range(num_variables)]
  upper_bound = [1.0 for i in range(num_variables)]

  # Computing the coefficients for objective function
  objective = cost_function(dataset, centres, num_variables, prohibited_assignments)

  return objective, lower_bound, upper_bound, variable_names, P, C




def prepare_to_add_constraints(dataset, centres, upper_cap, lower_cap, P, C, cons, fair_distance, ratings, flag, w1=0.5, w2=0.5):

  num_samples = len(dataset)
  num_centres = len(centres)

  rhs = []
  senses = []
  row_names = []
  coefficients = []
  eqn_id = 0				# Denotes the id of the constraint being processed currently

  distance = 0
  distance0 = L2Distance(dataset)
  distance1 = abs_difference(ratings)
  # distance2 = w1*distance0 + w2*distance1

  if flag==0:
    distance = distance0
  
  if flag==1:
    distance, fair_distance = minmax(distance1, fair_distance)
  
  if flag==2:
    distance2 = w1*minmax(distance0, 0)[0] + w2*minmax(distance1, 0)[0]
    distance, fair_distance = minmax(distance2, fair_distance)

  # Constraint type 1: Summation of P values over all clusters = 1 for each sample
  for point in range(num_samples):
    rhs.append(1.0)
    senses.append("E")
    row_names.append("{eqn_id}_Total_probability_{pt}".format(
        pt = point,
        eqn_id=eqn_id
      ))
    
    for centre in range(num_centres):
      coefficients.append((eqn_id, int(P[point][centre]), 1))

    eqn_id += 1
  
  num_probability_constraints = len(senses)
  print("Number of probability constraints:", num_probability_constraints)



  # Constraint type 2: Number of expected delivery agents assigned to ffc is lesser than max capacity of ffc
  for centre in range(num_centres):
    rhs.append(float(upper_cap[centre]))
    senses.append("L")
    row_names.append("{eqn_id}_Max_Cap_{centre}".format(
            eqn_id = eqn_id,
            centre = centre
          ))
    for point in range(num_samples):
      coefficients.append((eqn_id,int(P[point][centre]),1))

    eqn_id+=1

  num_maxcap_constraints = len(senses) - num_probability_constraints 
  print("Number of maximum capacity constraints:", num_maxcap_constraints)
  


  # Constraint type 3: Number of expected delivery agents assigned to ffc is greater than min capacity of ffc
  for centre in range(num_centres):
    rhs.append(float(lower_cap[centre]))
    senses.append("G")
    row_names.append("{eqn_id}_Min_Cap_{centre}".format(
						eqn_id = eqn_id,
						centre = centre
					))
    for point in range(num_samples):
      coefficients.append((eqn_id,int(P[point][centre]),1))

    eqn_id+=1

  num_mincap_constraints = len(senses) - (num_probability_constraints + num_maxcap_constraints)
  print("Number of minimum capacity constraints:", num_mincap_constraints) 


  
  # Constraint type 4: Lower and upper bound the abs values using their corresponding C variables
  for _point1 in range(num_samples):
    for _point2 in range(_point1+1,num_samples):
      for centre in range(num_centres):
        if distance[_point1][_point2]>=fair_distance:
          continue

        # Upper bound : -(P[i][k] - P[j][k]) <= C[i][j][k] <--> (P[i][k] - P[j][k]) <= -C[i][j][k]
        rhs.append(0)
        senses.append("L")
        row_names.append("{eqn_id}_Upper_Bound_{pt1}_{pt2}_{centre}".format(
            eqn_id = eqn_id,
            pt1 = _point1,
            pt2 = _point2,
            centre = centre
          ))
        coefficients.append((eqn_id, int(P[_point1][centre]), 1))
        coefficients.append((eqn_id, int(P[_point2][centre]), -1))
        coefficients.append((eqn_id, int(C[_point1][_point2][centre]), -1))

        eqn_id += 1

        # Lower_bound : (P[i][k] - P[j][k]) >= C[i][j][k] 
        rhs.append(0)
        senses.append("G")
        row_names.append("{eqn_id}_Lower_bound_{pt1}_{pt2}_{centre}".format(
            eqn_id = eqn_id,
            pt1 = _point1,
            pt2 = _point2,
            centre = centre
          ))
        coefficients.append((eqn_id, int(P[_point1][centre]), 1))
        coefficients.append((eqn_id, int(P[_point2][centre]), -1))
        coefficients.append((eqn_id, int(C[_point1][_point2][centre]), 1))

        eqn_id += 1

  num_lb_ub_constraints = len(senses) - (num_probability_constraints + num_mincap_constraints + num_maxcap_constraints) 
  print("Number of lower and upper bound constraints for abs constraints:", num_lb_ub_constraints)


  # Constraint type 5: Add fairness constraints in terms of C variables - trick to make mod constraints linear
  for _point1 in range(num_samples):
    for _point2 in range(_point1+1,num_samples):
      if distance[_point1][_point2] < fair_distance:
        rhs.append(cons * distance[_point1][_point2])		# Multiply by 2 to account for the division in TV distance expression
        senses.append("L")
        row_names.append("{eqn_id}_Fainess_{pt1}_{pt2}".format(
            eqn_id = eqn_id,
            pt1 = _point1,
            pt2 = _point2
          ))

        for centre in range(num_centres):
          coefficients.append((eqn_id, int(C[_point1][_point2][centre]), 1))
        # \sigma_k {C[u][v][1] + C[u][v][2] + ... + C[u][v][k]} \leq (d(u, v) * cons) 
        eqn_id += 1

  num_fairness_constraints = len(senses) - (num_probability_constraints + num_mincap_constraints + num_maxcap_constraints + num_lb_ub_constraints) 
  print("Number of fairness constraints:", num_fairness_constraints) # number of pairs of similar drivers considered
  # other way of getting the number of constraints is to use the 'row_names' 

  print("Total number of constraints:", len(senses))

  return rhs, senses, row_names, coefficients
