# baseball_shift

This program generates the baseball line-up to consider (1) players' strengths and (2) development opportunities for all players.

(1) It will consider the depth map of the players (that is, you think, where the players can play well, so-so, or not-well.)  For example, X is good at SS, OK at CF, but cannot be C. Y is good at C, 3B, and 1B, but not CF.
(2) It will consider any "must-have" assignments, e.g., the pitchers in the order of appearance (e.g., X, X, Y, Z, Z, Z, etc.), the catchers, etc.  You can give a short list of "must-have" assignments so that the program generates the list or a long list of "must-have" assignments so that the program only figures out the tedious assignments.  
(3) It will consider player-development rules, e.g., do you want all players to rotate every innings if possible, or do you want players to stay at the same positions for a couple of innings?  (For example, Pirates prefer all players to play different roles every innings, except catcher, so that a player can have a chance to develop their secondary/third positions.)
(4) It will consider preferred position-development rules, e.g., do you want a particular position to be played by one person for more than 3 or 4 innings?  Or do you want players to try the "hot" positions?  (For example, Pirates prefer no players to play SS for more than 2 innings during a game so that more players can get better as SS.)
(5) It will consider preferred "game-winning" starting assignment rules, e.g., for the highest chance to win a game, you prefer X play SS, Y play CF, in the first 4 or 5 innings of the game, while maintaining the opportunity for player development.  
