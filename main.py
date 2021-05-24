import ludopy
import random
import operator
import math
import numpy as np
import multiprocessing
import time
from matplotlib import pyplot as plt

class Agent:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness


def generate_random_population(pop_size, chromosome_size):

    random_population = []

    for i in range(pop_size):
        chromosome = []
        for j in range(chromosome_size):
            n = random.randint(-100,100)
            chromosome.append(n)
        agent = Agent(chromosome,0)
        random_population.append(agent)

    return random_population


def utility_helper(dice, player_pieces, enemy_pieces, piece):

    player_after_move = player_pieces[piece] + dice
    player_current_pos = player_pieces[piece]

    activation_vector = []

    # Offset the position of the enemies so that it is in the same frame as the player
    for i in range(len(enemy_pieces)):
        for j in range(len(enemy_pieces[i])):
            if enemy_pieces[i][j] < 54 and enemy_pieces[i][j] != 0:
                enemy_pieces[i][j] += ((i+1)*13)
                if enemy_pieces[i][j] >= 53:
                    enemy_pieces[i][j] = enemy_pieces[i][j] % 53 + 1

    # Going through each block to see if conditions for a state is met

    # State: The piece can be moved out of the home position
    if dice == 6 and player_current_pos == 0:
        activation_vector.append(1)
    else:
        activation_vector.append(0)

    # State: The piece can be moved into the end field
    if player_after_move == 59:
        activation_vector.append(1)
    else:
        activation_vector.append(0)

    # State: A piece can be moved within striking distance of an enemy
    activation_strike = False
    for i in range(3):
        for j in range(4):
            if player_after_move + 6 > enemy_pieces[i][j] > player_after_move:
                activation_vector.append(1)
                activation_strike = True
                break
    if not activation_strike: activation_vector.append(0)

    # State: A piece can hit another player home
    activation_hit = False
    hit_counter = 0
    for i in range(3):
        for j in range(4):
            if player_after_move == enemy_pieces[i][j]:
                hit_counter += 1
        if hit_counter == 1:
            activation_vector.append(1)
            activation_hit = True
        else:
            activation_vector.append(0)
            activation_hit = True
    if not activation_hit: activation_vector.append(0)

    # State: moving a piece onto another players starting position
    if player_after_move < 0:
        if player_after_move % 13 == 0:
            activation_vector.append(1)
        else:
            activation_vector.append(0)
    else:
        activation_vector.append(0)

    # State: moving a piece into danger
    activation_danger = False
    for i in range(3):
        for j in range(4):
            if player_after_move - 6 < enemy_pieces[i][j] < player_after_move:
                activation_vector.append(1)
                activation_danger = True
                break
        if activation_danger: break
    if not activation_danger: activation_vector.append(0)

    # State: Move onto star
    activation_star = False
    stars = [5, 12, 18, 25, 31, 38, 44, 51]
    for star in stars:
        if player_after_move == star:
            activation_vector.append(1)
            activation_star = True
            break
    if not activation_star: activation_vector.append(0)

    # State: Move onto globe
    activation_globe = False
    globes = [1, 9, 22, 35, 48]
    for globe in globes:
        if player_after_move == globe:
            for i in range(3):
                for j in range(4):
                    if player_after_move == enemy_pieces[i][j]:
                        activation_vector.append(0)
                        activation_globe = True
                        break
        if activation_globe: activation_vector.append(1)
        break
    if not activation_globe: activation_vector.append(0)
    return activation_vector


def utility_function(agent, dice, move_pieces, player_pieces, enemy_pieces):

    best_piece = move_pieces[0]
    piece_scores = []
    chromosome = agent.chromosome
    best_score = 0

    for piece in move_pieces:

        activation = utility_helper(dice, player_pieces, enemy_pieces, piece)

        piece_score = chromosome[0] * activation[0] + chromosome[1] * activation[1] + chromosome[2] * activation[2] \
                      + chromosome[3] * activation[3] + chromosome[4] * activation[4] + chromosome[5] * activation[5] \
                      + chromosome[6] * activation[6] + chromosome[7] * activation[7]

        piece_scores.append(piece_score)

        if piece_score > best_score:
            best_score = piece_score
            best_piece = piece

    # add in a way to make sure that the first one is no negative
    for i in range(len(piece_scores)):
        pass

    return best_piece


def simulate_ludo_games(agent, number_of_games, return_value):

    wins = 0

    for i in range(number_of_games):
        g = ludopy.Game()
        there_is_a_winner = False

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if player_i == 0:
                    piece_to_move = move_pieces[(utility_function(agent, dice, move_pieces,player_pieces,enemy_pieces))%len(move_pieces)]
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

        if player_i == 0:
            wins += 1
    return_value.value = wins / number_of_games * 100
    return wins / number_of_games * 100


def simulate_population(population, amount_of_games):
    best_chromosome = population[0]
    ret_value = multiprocessing.Value("d", 0.0, lock=False)

    for i in range(len(population)):
        print("Testing Fitness of Chromosome " + str(i))
        population[i].fitness = simulate_ludo_games(population[i], amount_of_games, ret_value)
        #print("Winrate: " + str(population[i].fitness))

        if population[i].fitness > best_chromosome.fitness:
            best_chromosome = population[i]

    #print("Best Chromosome Fitness " + str(best_chromosome.fitness))

    # Population becomes sorted from the worst fitness to the best
    sorted_population = sorted(population, key=operator.attrgetter("fitness"))

    return sorted_population


def tournament_selection(population_, tournament_size):
    selected_parents = []

    for i in range(math.ceil(len(population_)/2)):
        parent_pair = []
        for j in range(2):
            tournament_participants = []
            for k in range(tournament_size):
                participant = population_[np.random.randint(0, len(population_))]
                tournament_participants.append(participant)
            best_participant = tournament_participants[0]
            for n in range(len(tournament_participants)):
                if tournament_participants[n].fitness > best_participant.fitness:
                    best_participant = tournament_participants[n]
            parent_pair.append(best_participant)
        selected_parents.append(parent_pair)

    return selected_parents


def crossover(parents_, amount_of_children):
    children = []

    for i in range(len(parents_)):
        # load in the chromosomes of both parents
        chromosome_one = parents_[i][0].chromosome
        chromosome_two = parents_[i][1].chromosome

        #single point crossover
        for j in range(amount_of_children):
            crossover_point = np.random.randint(0, len(chromosome_one))
            new_chromosome = []
            for k in range(crossover_point):
                new_chromosome.append(chromosome_one[k])
            for l in range(len(chromosome_one)-crossover_point):
                new_chromosome.append(chromosome_two[l+crossover_point])

            child = Agent(new_chromosome,0)
            children.append(child)

    return children


def mutate(children, mutation_rate, mutation_strength):
    for i in range(len(children)):
        mutated_chromosome = children[i].chromosome
        for j in range(len(mutated_chromosome)):
            mutation_chance = np.random.randint(0, 100)
            if mutation_chance <= mutation_rate:
                mutation = np.random.randint(-mutation_strength, mutation_strength)
                mutated_chromosome[j] += mutation
        m = np.array(mutated_chromosome)
        np.clip(m,-100,100, m)
        children[i].chromosome = m.tolist()


def steady_state(population_, children_, population_size_):
    new_population_ = []
    combined_pop = population_ + children_
    sorted_combined_pop = sorted(combined_pop, key=operator.attrgetter("fitness"))
    for i in range(len(sorted_combined_pop), len(sorted_combined_pop) - population_size_, -1):
        new_population_.append(sorted_combined_pop[i-1])
    return new_population_


def simulate_generations(starting_generation, number_of_generations, win_rate_stopping_criteria, games_played_to_test_fitness, pop_size):
    generation_counter = 0
    generation = starting_generation
    best_win_rate = 0

    lowest_win_rate_list = []
    avg_win_rate_list = []
    highest_win_rate_list = []

    best_chromosome = generation[-1]

    while generation_counter <= number_of_generations and best_win_rate < win_rate_stopping_criteria:
        print("Training Generation " + str(generation_counter))

        lowest_win_rate = generation[0].fitness
        lowest_win_rate_list.append(lowest_win_rate)
        average_win_rate = 0
        highest_win_rate = generation[-1].fitness
        for pop in generation:
            average_win_rate += pop.fitness
        average_win_rate = average_win_rate / len(generation)
        avg_win_rate_list.append(average_win_rate)
        highest_win_rate_list.append(highest_win_rate)

        print("Lowest Win Rate: " + str(lowest_win_rate) + " Average Win Rate: " + str(average_win_rate) + " Highest Win Rate: " + str(highest_win_rate))

        # Selection Step
        best_half_of_generation = generation[int(len(generation) / 2):len(generation)]
        parents = tournament_selection(best_half_of_generation, 3)

        # Crossover and Mutation step
        offspring = crossover(parents, 16)
        mutate(offspring, 20, 10)

        # Add new blood to the mix
        new_blood = generate_random_population(25, 8)
        next_gen = offspring + new_blood

        # Test offspring
        tested_offspring = simulate_population(next_gen, games_played_to_test_fitness)

        # Steady state step
        generation = steady_state(best_half_of_generation, tested_offspring, pop_size)

        generation_counter += 1
        sorted_generation = sorted(generation, key=operator.attrgetter("fitness"))
        best_win_rate = sorted_generation[-1].fitness
        best_chromosome = sorted_generation[-1].chromosome
        print("Best win rate so far: " + str(best_win_rate))
        generation = sorted_generation

    return lowest_win_rate_list, avg_win_rate_list, highest_win_rate_list, best_chromosome


def plot_tests(low, avg, high, gen_size):
    gen = list(range(1, gen_size + 2))

    p1 = plt.plot(gen,low, '-r')
    p2 = plt.plot(gen, avg, '-g')
    p3 = plt.plot(gen, high, '-b')

    plt.xlabel('Generations')
    plt.ylabel('Win rate')
    plt.legend((p3[0], p2[0], p1[0]), ('Highest', 'Average', 'Lowest'), loc ="lower right")

    plt.show()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    population_size = 75
    # Generate random population and test their fitness
    population = generate_random_population(population_size,8)

    initial_tested_generation = simulate_population(population, 100)

    gen_count = 25

    low, avg, high, best_chrome = simulate_generations(initial_tested_generation,gen_count,80,100,population_size)

    print("Best Chromosome: " + str(best_chrome))

    plot_tests(low, avg, high, gen_count)
