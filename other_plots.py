# create plots to understand CCGP and PS values


def p_score2(data, labels, dichotomy=([1, 2, 3, 4], [5, 6, 7, 8])):
    
    """Returns the 24 cosines for the all one-to-one possible digit pairings of the dichotomy (rather than just the max cosine), 
    as well as labels for the two pairings making up each angle.
    
	# Arguments 
            data (array): Input data with two dimensional shape (T*C, N).
            labels (array): Corresponding condition labels with same length as input data (N).
            dichotomy (tuple): Tuple of two int arrays representing two sides of a dichotomy.
    
        # Returns 
            cosines (array): Array with 24 cosines of the hyperplane angle for each pairing
            labels (array): Array with 24 tuples containing the 2 pairings corresponding to the angle
            [format: (pairing 1, pairing 2)]
    
    """  

    cosines = np.zeros((24, 6))
    total_labels = []

    label1b = dichotomy[1][0]
    label2b = dichotomy[1][1]
    label3b = dichotomy[1][2]
    label4b = dichotomy[1][3]

    index = 0
    for perm in permutations(dichotomy[0]):

        label1a = perm[0]
        target1 = []
        data1 = []

        label2a = perm[1]
        target2 = []
        data2 = []

        label3a = perm[2]
        target3 = []
        data3 = []

        label4a = perm[3]
        target4 = []
        data4 = []

        for lab in range(len(labels)):

            label = labels[lab]

            if not isinstance(label, float):
                print(type(label))
                break

            if int(labels[lab]) == label1a:
                # if np.isclose(labels[lab], label1a, rtol=1e-05, atol=1e-08, equal_nan=False):

                target1.append(0)

                data1.append(data[lab])

            if int(labels[lab]) == label1b:

                target1.append(1)

                data1.append(data[lab])

            if int(labels[lab]) == label2a:

                target2.append(0)

                data2.append(data[lab])

            if int(labels[lab]) == label2b:

                target2.append(1)

                data2.append(data[lab])

            if int(labels[lab]) == label3a:

                target3.append(0)

                data3.append(data[lab])

            if int(labels[lab]) == label3b:

                target3.append(1)

                data3.append(data[lab])

            if int(labels[lab]) == label4a:

                target4.append(0)

                data4.append(data[lab])

            if int(labels[lab]) == label4b:

                target4.append(1)

                data4.append(data[lab])

        target1 = np.array(target1)
        target2 = np.array(target2)
        target3 = np.array(target3)
        target4 = np.array(target4)

        data1 = np.array(data1)
        data2 = np.array(data2)
        data3 = np.array(data3)
        data4 = np.array(data4)

        svc1 = LinearSVC(C=1.0, max_iter=50000)

        svc1.fit(data1, target1.T)

        vector1 = svc1.coef_[0]

        svc2 = LinearSVC(C=1.0, max_iter=50000)

        svc2.fit(data2, target2.T)

        vector2 = svc2.coef_[0]

        svc3 = LinearSVC(C=1.0, max_iter=50000)

        svc3.fit(data3, target3.T)

        vector3 = svc3.coef_[0]

        svc4 = LinearSVC(C=1.0, max_iter=50000)

        svc4.fit(data4, target4.T)

        vector4 = svc4.coef_[0]

        avg_cosine = np.ones(6)

        cosine_labels = []

        avg_cosine[0] = cosine(vector1, vector2)
        # vector1 is labels 1a,1b
        # vector2 is labels 2a,2b
        cosine_labels.append(((label1a, label1b), (label2a, label2b)))

        avg_cosine[1] = cosine(vector1, vector3)
        cosine_labels.append(((label1a, label1b), (label3a, label3b)))

        avg_cosine[2] = cosine(vector1, vector4)
        cosine_labels.append(((label1a, label1b), (label4a, label4b)))

        avg_cosine[3] = cosine(vector2, vector3)
        cosine_labels.append(((label2a, label2b), (label3a, label3b)))

        avg_cosine[4] = cosine(vector2, vector4)
        cosine_labels.append(((label2a, label2b), (label4a, label4b)))

        avg_cosine[5] = cosine(vector3, vector4)
        cosine_labels.append(((label3a, label3b), (label4a, label4b)))

        cosines[index] = avg_cosine

        total_labels.append(cosine_labels)

        index += 1

    return cosines, total_labels


def ccgp2(data, labels, dichotomy=([1, 2, 3, 4], [5, 6, 7, 8])):
    
    """Returns all 16 classification accuracies that make up the CCGP for a given dichotomy
    (as opposed to just the average accuracy)
    
	# Arguments 
            data (array): Input data with two dimensional shape (T*C, N).
            labels (array): Corresponding condition labels with same length as input data (N).
            dichotomy (tuple): Tuple of two int arrays representing two sides of a dichotomy.
    
    # Returns 
            accuracy (array): 16 test classification accuracies for each of the training sets possible per dichotomy
    
    """  
    labels1 = dichotomy
    label_1a = dichotomy[0]
    label_1b = dichotomy[1]

    accuracy = []

    label_1a_combs = combinations(label_1a, 3)
    label_1b_combs = combinations(label_1b, 3)

    label_1a_set = []
    label_1b_set = []

    for a in label_1a_combs:

        label_1a_set.append(a)

    for b in label_1b_combs:

        label_1b_set.append(b)

    for train_a in label_1a_set:

        for train_b in label_1b_set:

            label_2a = labels1[0]  # dichotomy a
            label_2b = labels1[1]  # dichotomy b

            target1 = []
            target2 = []

            data_train = []

            data_test = []

            for lab in range(len(labels)):
                if int(labels[lab]) in label_2a:  # [1,2,3,4]
                    if int(labels[lab]) in train_a:  # [1,2,3]

                        target1.append(0)

                        data_train.append(data[lab])

                    else:
                        target2.append(0)  # [4]

                        data_test.append(data[lab])

                else:  # [5,6,7,8]

                    if int(labels[lab]) in train_b:

                        target1.append(1)
                        data_train.append(data[lab])

                    else:
                        target2.append(1)

                        data_test.append(data[lab])

            target1 = np.array(target1)
            target2 = np.array(target2)

            data_train = np.array(data_train)
            data_test = np.array(data_test)

            # fit on training data
            svc1 = LinearSVC(C=1.0, max_iter=50000)

            svc1.fit(data_train, target1.T)

            # test on testing data
            acc = svc1.score(data_test, target2.T)
            accuracy.append(acc)

    return accuracy


def jitter_x(index, data_length):
    
    """Returns jittered coordinates around a certain value, in order to jitter points 
    horizontally or vertically on scatter plot
    
	# Arguments 
            index (double): index around which to jitter values
            data_length (int): number of values to jitter
            
        # Returns 
            new_data (array): jittered values of specified length
    
    """  
    
    new_data = []

    for i in range(data_length):

        new_data.append(np.random.normal(index, 0.05))

    return new_data


def create_PS_plots(data):
    
    """35 plots, one per dichotomy, to show spread of 6 cosines across each of 24 pairings for parallelism score
    (red star = average cosine across pairing)
    
	# Arguments 
            data (array): Input data with two dimensional shape (T*C, N).
    
    #Returns
            None
    
    """  

    for d in dichotomies:

        cosines, label = p_score2(data, new_labels, d)

        plt.figure(figsize=(20, 10))
        for i in range(24):
            jitter = jitter_x(i, 6)

            plt.scatter(jitter, cosines[i], s=50)

            plt.scatter(i, np.mean(cosines[i]), c="red", marker="*", s=300)

        plt.ylabel("Cosine")
        plt.xlabel("Angle Pairings")
        plt.title("{}".format(str(d)))
        plt.show()


def create_CCGP_plot(data_reshaped):
 
    """One plot to show spread of 16 classification accuracies across each dichotomy for CCGP
    (red star = average accuracy) 
    
	# Arguments 
            data (array): Input data with two dimensional shape (T*C, N).
    
    #Returns
            None
    
    """  
 
    index = 0

    plt.figure(figsize=(20, 10))
    plt.ylabel("CCGP Values")
    plt.xlabel("Pairings")

    for d in dichotomies:
        ccgp_values = ccgp2(data_reshaped, new_labels, d)

        jitter = jitter_x(index, 16)

        plt.scatter(jitter, ccgp_values)

        plt.scatter(index, np.mean(ccgp_values), c="red", marker="*", s=300)

        index += 1

    plt.title("Real CCGP")
    plt.show()


if __name__ == "__main__":

    NN_data = np.load("./values/new_NN_data.npy")

    create_PS_plots(NN_data)
    create_CCGP_plot(NN_data)
