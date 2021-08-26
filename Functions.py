from mplsoccer import Pitch
import matplotlib.pyplot as plt
from mplsoccer.statsbomb import read_event, EVENT_SLUG

# get match dataframe
MATCH_ID = '22912'
match_json = f'{EVENT_SLUG}/{MATCH_ID}.json'
event_dict = read_event(match_json)
event_df = event_dict['event']
players_df = event_dict['tactics_lineup']


def get_teams_name():
    # get match teams name
    team1, team2 = event_df.team_name.unique()
    return team1, team2


def get_pass_df(team):
    # boolean mask if the event is pass and team is our team
    team_pass_mask = (event_df.type_name == 'Pass') & (event_df.team_name == team)
    # create a dataframe of the passes only
    pass_df = event_df[team_pass_mask]
    return pass_df


def pass_arrows(team):
    pass_df = get_pass_df(team)
    # boolean mask if the pass is completed
    complete_pass_mask = pass_df.outcome_name.isnull()
    completed_passes = pass_df[complete_pass_mask]
    Incomplete_passes = pass_df[~complete_pass_mask]
    # setup the pitch
    pitch = Pitch(pitch_color='grass', line_color='white')
    fig, ax = pitch.draw()
    # plot the completed passes arrows
    pitch.arrows(
        xstart=completed_passes.x, ystart=completed_passes.y, xend=completed_passes.end_x,
        yend=completed_passes.end_y, width=1, ax=ax, headwidth=10, headlength=10, color='#cccc00',
        label='Completed Passes')
    # plot the Incomplete passes arrows
    pitch.arrows(
        xstart=Incomplete_passes.x, ystart=Incomplete_passes.y, xend=Incomplete_passes.end_x,
        yend=Incomplete_passes.end_y, width=0.8, ax=ax, headwidth=10, headlength=10, color='#990000',
        label='Incomplete Passes')
    # set the title
    ax.set_title(f'{team} Passes')
    # set the legend
    ax.legend(loc='upper left')
    plt.show()


