#!/usr/bin/env python3
# Copyright 2010-2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 Yen-Kuang Chen
"""Creates a shift scheduling problem and solves it."""

from array import *
from absl import app
from absl import flags

from google.protobuf import text_format
from ortools.sat.python import cp_model

#importing the required libraries
import gspread
from gspread.cell import Cell
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import numpy
import os, time
import logging, sys

#JSON_FILE_PATH='/root/baseball_shift/'
JSON_FILE_PATH='/home/pallgcsk/baseball_shift/'
#JSON_FILE_PATH='/home/ykchen/baseball_shift/'

TEAM_LINEUP='AkitaLineup'
#TEAM_LINEUP='BaseballLineup'

DEBUG_INFO_LEVEL=logging.WARNING
#DEBUG_INFO_LEVEL=logging.INFO

flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:15.0',
                    'Sat solver parameters.')
FLAGS = flags.FLAGS 
FLAGS(sys.argv)

def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.

  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.

  Args:
    works: a list of variables to extract the span from.
    start: the start to the span.
    length: the length of the span.

  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence


def add_soft_sequence_constraint(model, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """Sequence constraint on true variables with soft and hard bounds.

  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.

  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    prefix: a base name for penalty literals.

  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Penalize sequences that are below the soft limit.
    if min_cost > 0:
        for length in range(hard_min, soft_min):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': under_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # We filter exactly the sequence with a short length.
                # The penalty is proportional to the delta with soft_min.
                cost_coefficients.append(min_cost * (soft_min - length))

    # Penalize sequences that are above the soft limit.
    if max_cost > 0:
        for length in range(soft_max + 1, hard_max + 1):
            for start in range(len(works) - length + 1):
                span = negated_bounded_span(works, start, length)
                name = ': over_span(start=%i, length=%i)' % (start, length)
                lit = model.NewBoolVar(prefix + name)
                span.append(lit)
                model.AddBoolOr(span)
                cost_literals.append(lit)
                # Cost paid is max_cost * excess length.
                cost_coefficients.append(max_cost * (length - soft_max))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])
    return cost_literals, cost_coefficients


def add_soft_sum_constraint(model, works, hard_min, soft_min, min_cost,
                            soft_max, hard_max, max_cost, prefix):
    """Sum constraint with soft and hard bounds.

  This constraint counts the variables assigned to true from works.
  If forbids sum < hard_min or > hard_max.
  Then it creates penalty terms if the sum is < soft_min or > soft_max.

  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a sum of at least
      hard_min.
    soft_min: any sequence should have a sum of at least soft_min, or a linear
      penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the sum is less than
      soft_min.
    soft_max: any sequence should have a sum of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a sum of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the sum is more than
      soft_max.
    prefix: a base name for penalty variables.

  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_variables = []
    cost_coefficients = []
    sum_var = model.NewIntVar(hard_min, hard_max, '')
    # This adds the hard constraints on the sum.
    model.Add(sum_var == sum(works))

    # Penalize sums below the soft_min target.
    if soft_min > hard_min and min_cost > 0:
        delta = model.NewIntVar(-len(works), len(works), '')
        model.Add(delta == soft_min - sum_var)
        # TODO(user): Compare efficiency with only excess >= soft_min - sum_var.
        excess = model.NewIntVar(0, 7, prefix + ': under_sum')
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(min_cost)

    # Penalize sums above the soft_max target.
    if soft_max < hard_max and max_cost > 0:
        delta = model.NewIntVar(-7, 7, '')
        model.Add(delta == sum_var - soft_max)
        excess = model.NewIntVar(0, 7, prefix + ': over_sum')
        model.AddMaxEquality(excess, [delta, 0])
        cost_variables.append(excess)
        cost_coefficients.append(max_cost)

    return cost_variables, cost_coefficients


def solve_shift_scheduling(params, output_proto):
    """Solves the shift scheduling problem."""

    logging.basicConfig(stream=sys.stderr, level=DEBUG_INFO_LEVEL)
    logger = logging.getLogger("Baseball_Shift")

    # define the scope
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

    # add credentials to the account
    creds = ServiceAccountCredentials.from_json_keyfile_name(JSON_FILE_PATH+'baseballlineup-d679cf590579.json', scope)

    # authorize the clientsheet
    client = gspread.authorize(creds)

    # get the instance of the Spreadsheet
    sheet = client.open(TEAM_LINEUP)

    # Data
    names=sheet.worksheet('BattingOrder').col_values(1)
    num_players = len(names)
    # batting_order = [0...num_players-1]
    num_games = 1

    depth_matrix=sheet.worksheet('DepthMatrix').get_all_values()
    shifts=depth_matrix.pop(0)
    del shifts[0]
    #shifts = ['DH', 'P ', 'C ', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
    #         '0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9'

    # Depth map: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    transposed_depth_chart=numpy.transpose(depth_matrix)
    depth_chart_list=list(transposed_depth_chart[0])

    # Shift constraints on continuous sequence :
    #     (shift, hard_min, soft_min, min_penalty,
    #             soft_max, hard_max, max_penalty)
    shift_constraints = [
        # No two consecutive innings for DH 
        (0, 1, 1, 0, 1, 7, 10),
        # No two consecutive innings for same IF position 
        (3, 1, 1, 0, 1, 1, 0),
        (4, 1, 1, 0, 1, 1, 0),
        (5, 1, 1, 0, 1, 1, 0),
        (6, 1, 1, 0, 1, 1, 0),
        # No three consecutive innings for same OF position 
        (7, 1, 1, 0, 1, 2, 0),
        (8, 1, 1, 0, 1, 2, 0),
        (9, 1, 1, 0, 1, 2, 0),
    ]

    # Game sum constraints on shifts innings:
    #     (shift, hard_min, soft_min, min_penalty,
    #             soft_max, hard_max, max_penalty)
    game_sum_constraints = [
        # Constraints on positions per game.
        (0, 0, 1, 7, 2, 7, 4), # For 12 players roster, one should not be DH for 3 of more innings
        (1, 0, 0, 7, 1, 1, 4),
        (2, 0, 0, 7, 2, 3, 4), # if necessary, one can catch for 3 innings
        (3, 0, 0, 7, 2, 3, 4), # limit players on same positions for more than twice
        (4, 0, 0, 7, 2, 2, 4),
        (5, 0, 0, 7, 2, 2, 4),
        (6, 0, 0, 7, 2, 2, 4),
        (7, 0, 0, 7, 2, 3, 4), # if necessary, one can OF for 3 innings
        (8, 0, 0, 7, 2, 3, 4),
        (9, 0, 0, 7, 2, 3, 4),
    ]

    # Penalized transitions:
    #     (previous_shift, next_shift, penalty (0 means forbidden))
    penalized_transitions = [
        # After pitcher/catcher to catcher/pitcher is not preferred 
        #(1, 2, 10),
        #(2, 1, 10),
    ]

    # Demands for shifts for each inning of the game.
    game_cover_demands = [
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 1st // DH is implicitly not required
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 2nd
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 3rd
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 4th
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 5th 
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 6th 
        (1, 1, 1, 1, 1, 1, 1, 1, 1),  # 7th 
    ]

    # Penalty for exceeding the cover constraint per shift type.
    excess_cover_penalties = (5, 5, 5, 5, 5, 5, 5, 5, 5) # DH is implicity not required

    num_innings = num_games * 7
    num_shifts = len(shifts)

    model = cp_model.CpModel()

    work = {}
    for e in range(num_players):
        for s in range(num_shifts):
            for d in range(num_innings):
                work[e, s, d] = model.NewBoolVar('work%i_%i_%i' % (e, s, d))

    # Linear terms of the objective in a minimization context.
    obj_int_vars = []
    obj_int_coeffs = []
    obj_bool_vars = []
    obj_bool_coeffs = []

    # Exactly one shift per inning.
    for e in range(num_players):
        for d in range(num_innings):
            model.AddExactlyOne(work[e, s, d] for s in range(num_shifts))

    # Fixed assignments.
    # Fixed assignment: (player, shift, inning).
    # player starts with 0, innings starts with 0
    # shift starts with 1 (pitcher)
    logger.info ("Fixed Assignment")
    fixed_assignment=sheet.worksheet('FixedAssignment').get_all_values()
    del fixed_assignment[0]
    for assignment in fixed_assignment:
        player_index= names.index(assignment[0])
        shift_index = shifts.index(assignment[1])
        inning_index = int(assignment[2])-1
        logger.info("%s %d %s %d %d %d" % (assignment[0], player_index, assignment[1], shift_index, int(assignment[2]), inning_index))
        model.Add(work[player_index, shift_index, inning_index] == 1)

    logger.info ("")

    # Pitcher assignment // special case of fixed assignment, where the shift_index = 1
    logger.info ("Pitching Order")
    pitcher_list=sheet.worksheet('PitchingOrder').col_values(1)
    for pitcher in pitcher_list:
        player_index= names.index(pitcher)
        inning_index = pitcher_list.index(pitcher)
        logger.info("%s %d 1 %d" % (pitcher, player_index, inning_index))
        model.Add(work[player_index, 1, inning_index] == 1)

    logger.info ("")

    # Player depth map 
    """ an example of target depth_matrix
    depth_map = [
        [ 0, 1, 0, 0, 1, 0, 2, 0, 0, -1 ],      # 0 
        [ 0, 2, 1, 0, 2, 0, 2, -1, 1, 0 ],      # 1 
        [ 0, 1, 4, -1, 0, 0, 0, 0, -1, 0 ],      # 2
        [ 0, 2, 3, 3, 0, 1, 0, -1, -1, -1 ],    # 3 
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1 ],      # 4 
        [ 0, 0, -1, -1, 0, 0, 0, 1, 1, 1 ],     # 5 
        [ 0, 1, -1, -1, 0, 1, 0, 1, 1, 0 ],     # 6 
        [ 0, -1, -1, -1, 2, 0, -1, 1, -1, 1 ],   # 7
        [ 0, 2, 0, -1, 0, 1, 2, 0, 1, 0 ],      # 8 
        [ 0, 2, -1, 0, 0, 2, 0, 0, -1, 1 ],      # 9 
        [ 0, 1, -1, 3, -1, -1, -1, 0, -1, 1 ],   # 10 
        [ 0, 0, 0, -1, 2, -1, 1, 1, 1, 0 ]       # 11 
    ]
    """
    logger.info ("Depth Matrix by batting order")

    for player in names:
        player_index_in_matrix= depth_chart_list.index(player)
        player_index_in_batting_order= names.index(player)
        logger.info(depth_matrix[player_index_in_matrix])
        for s in range(num_shifts): 
            w = int(depth_matrix[player_index_in_matrix][s+1])
            if w > 0: 
                for d in range(num_innings):
                    obj_bool_vars.append(work[player_index_in_batting_order, s, d])
                    obj_bool_coeffs.append((-2)*w)
                    #logger.info ("%d %d %d %d" % (player_index_in_batting_order, s, d, w))
            elif w < 0:
                for d in range(num_innings):
                    obj_bool_vars.append(work[player_index_in_batting_order, s, d])
                    obj_bool_coeffs.append((-4)*w)
                    #logger.info ("%d %d %d %d" % (player_index_in_batting_order, s, d, w))

    logger.info ("")

    # Starting request: (player, shift, weight) // Only for the first 5 innings
    # A negative weight indicates that the player desire this assignment.
    # A postive weight indicates that the player does not desire this assignment.
    logger.info ("Starting Request")
    starting_request=sheet.worksheet('SRequest').get_all_values()
    del starting_request[0]
    for request in starting_request:
        player_index= names.index(request[0])
        shift_index = shifts.index(request[1])
        logger.info("%s %d %s %d %d" % (request[0], player_index, request[1], shift_index, int(request[2])))
        for d in range(5): # first 5 innings
            obj_bool_vars.append(work[player_index, shift_index, d])
            obj_bool_coeffs.append(-int(request[2])*2)

    logger.info ("")

    # Game request: (player, shift, weight) // For the whole game
    # A negative weight indicates that the player desire this assignment.
    # A postive weight indicates that the player does not desire this assignment.
    logger.info ("Game Request")
    game_request=sheet.worksheet('GRequest').get_all_values()
    del game_request[0]
    for request in game_request:
        player_index= names.index(request[0])
        shift_index = shifts.index(request[1])
        logger.info("%s %d %s %d %d" % (request[0], player_index, request[1], shift_index, int(request[2])))
        for d in range(num_innings):
            obj_bool_vars.append(work[player_index, shift_index, d])
            obj_bool_coeffs.append(-int(request[2])*2)

    logger.info ("")

    # Shift constraints
    for ct in shift_constraints:
        shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = ct
        for e in range(num_players):
            works = [work[e, shift, d] for d in range(num_innings)]
            variables, coeffs = add_soft_sequence_constraint(
                model, works, hard_min, soft_min, min_cost, soft_max, hard_max,
                max_cost,
                'shift_constraint(player %s, shift %i)' % (names[e], shift))
            obj_bool_vars.extend(variables)
            obj_bool_coeffs.extend(coeffs)

    # Game sum constraints
    for ct in game_sum_constraints:
        shift, hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = ct
        for e in range(num_players):
            for w in range(num_games):
                works = [work[e, shift, d + w * 7] for d in range(7)]
                variables, coeffs = add_soft_sum_constraint(
                    model, works, hard_min, soft_min, min_cost, soft_max,
                    hard_max, max_cost,
                    'game_sum_constraint(player %s, shift %i, game %i)' %
                    (names[e], shift, w))
                obj_int_vars.extend(variables)
                obj_int_coeffs.extend(coeffs)

    # Penalized transitions
    for previous_shift, next_shift, cost in penalized_transitions:
        for e in range(num_players):
            for d in range(num_innings - 1):
                transition = [
                    work[e, previous_shift, d].Not(), work[e, next_shift,
                                                           d + 1].Not()
                ]
                if cost == 0:
                    model.AddBoolOr(transition)
                else:
                    trans_var = model.NewBoolVar(
                        'transition (player=%s, inning=%i)' % (e, d))
                    transition.append(trans_var)
                    model.AddBoolOr(transition)
                    obj_bool_vars.append(trans_var)
                    obj_bool_coeffs.append(cost)

    # Cover constraints
    for s in range(1, num_shifts):
        for w in range(num_games):
            for d in range(7):
                works = [work[e, s, w * 7 + d] for e in range(num_players)]
                # Ignore Off shift.
                min_demand = game_cover_demands[d][s - 1]
                worked = model.NewIntVar(min_demand, num_players, '')
                model.Add(worked == sum(works))
                over_penalty = excess_cover_penalties[s - 1]
                if over_penalty > 0:
                    name = 'excess_demand(shift=%i, game=%i, inning=%i)' % (s, w,
                                                                         d)
                    excess = model.NewIntVar(0, num_players - min_demand,
                                             name)
                    model.Add(excess == worked - min_demand)
                    obj_int_vars.append(excess)
                    obj_int_coeffs.append(over_penalty)

    # Objective
    model.Minimize(
        sum(obj_bool_vars[i] * obj_bool_coeffs[i]
            for i in range(len(obj_bool_vars))) +
        sum(obj_int_vars[i] * obj_int_coeffs[i]
            for i in range(len(obj_int_vars))))

    if output_proto:
        logger.info('Writing proto to %s' % output_proto)
        with open(output_proto, 'w') as text_file:
            text_file.write(str(model))

    # Solve the model.
    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.Solve(model, solution_printer)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logger.info('')
        logger.info('Penalties:')
        for i, var in enumerate(obj_bool_vars):
            if solver.BooleanValue(var):
                penalty = obj_bool_coeffs[i]
                if penalty > 0:
                    logger.info('  %s violated, penalty=%i' % (var.Name(), penalty))
                else:
                    logger.info('  %s fulfilled, gain=%i' % (var.Name(), -penalty))

        for i, var in enumerate(obj_int_vars):
            if solver.Value(var) > 0:
                logger.info('  %s violated by %i, linear penalty=%i' % (var.Name(), solver.Value(var), obj_int_coeffs[i]))

        print('')
        print('<table>')
        header='<tr><td>         </td> '
        for d in range(num_innings):
            header += '<td> '+ str(d+1) + '  </td>'
        print(header+ ' </tr>')
        for e in range(num_players):
            schedule = ''
            for d in range(num_innings):
                for s in range(num_shifts):
                    if solver.BooleanValue(work[e, s, d]):
                        schedule += '<td> '+'{:2}'.format(shifts[s]) + ' </td>'
            print('<tr><td>%8s </td> %s </tr>' % (names[e], schedule))
        print('</table>')

        cells = []
        os.environ['TZ'] = 'America/Los_Angeles'
        time.tzset()
        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m/%d %H:%M", named_tuple)
        cells.append(Cell(row=1, col=1, value=time_string))

        for d in range(num_innings):
            cells.append(Cell(row=1, col=d+2, value=d+1))
        for e in range(num_players):
            cells.append(Cell(row=e+2, col=1, value=names[e]))
            for d in range(num_innings):
                for s in range(num_shifts):
                    if solver.BooleanValue(work[e, s, d]):
                        cells.append(Cell(row=e+2, col=d+2, value=shifts[s]))
        logger.info(cells)
        output_worksheet = sheet.worksheet('Lineup')
        output_worksheet.clear()
        output_worksheet.update_cells(cells)

    print()
    print('Statistics')
    print('  - status          : %s' % solver.StatusName(status))
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())


### Main ###
solve_shift_scheduling(FLAGS.params, FLAGS.output_proto)
