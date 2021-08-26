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


def pass_flow(team):
    pass_df = get_pass_df(team)
    # setup the pitch
    pitch = Pitch(pitch_color='grass', line_color='white', line_zorder=2)
    fig, ax = pitch.draw()
    # plot the heatmap (darker color means more passes from that square)
    bins = (6, 3)  # divide pitch to 6 columns and 3 rows
    bins_heatmap = pitch.bin_statistic(pass_df.x, pass_df.y, statistic='count', bins=bins)
    pitch.heatmap(bins_heatmap, ax=ax, cmap='Reds')
    # plot the arrows pass flow
    pitch.flow(
        xstart=pass_df.x, ystart=pass_df.y, xend=pass_df.end_x, yend=pass_df.end_y,
        ax=ax, color='black', arrow_type='same', arrow_length=5, bins=bins
    )
    ax.set_title(f'{team} Pass Flow Map')
    plt.show()


def pass_network(team):
    pass_df = get_pass_df(team)
    # boolean mask if the pass is completed
    complete_pass_mask = pass_df.outcome_name.isnull()
    completed_pass = pass_df[complete_pass_mask]
    # get first substitution minute
    subs = event_df[event_df.type_name == 'Substitution']
    first_sub_minute = subs['minute'].min()
    print(first_sub_minute)
    # passes before first sub
    completed_pass = completed_pass[completed_pass['minute'] < first_sub_minute]
    # players average passes location x, y , player_id => index
    avg_loc = completed_pass.groupby('player_id').agg({'x': ['mean'], 'y': ['mean']})
    avg_loc.columns = ['x', 'y']
    # pass between player_id and pass_recipient_id
    pass_between = completed_pass.groupby(['player_id', 'pass_recipient_id'], as_index=False).id.count()
    pass_between.columns = ['passer', 'recipient', 'passes_between']
    # merge avg_loc table with pass_between table throw passer column to get start x, y
    pass_between = pass_between.merge(avg_loc, left_on='passer', right_index=True)
    # merge avg_loc table with pass_between table throw recipient column to get end x, y
    pass_between = pass_between.merge(avg_loc, left_on='recipient', right_index=True, suffixes=['', '_end'])
    # setup the pitch
    pitch = Pitch(pitch_color='grass', line_color='white')
    fig, ax = pitch.draw()
    # plot passes network
    pitch.arrows(
        xstart=pass_between.x, ystart=pass_between.y, xend=pass_between.x_end,
        yend=pass_between.y_end, width=1, headwidth=10, headlength=30, color='#990000', ax=ax)
    # plot players
    pitch.scatter(
        x=pass_between.x, y=pass_between.y, s=250, color='red', edgecolor='black', linewidth=1, alpha=1, ax=ax)
    # plot players jersey number
    for index, row in avg_loc.iterrows():
        # index in avg_loc table is the player_id
        player_jersey_number = players_df[players_df.player_id == index].player_jersey_number.values[0]
        pitch.annotate(player_jersey_number, xy=(row.x, row.y), c='white', va='center', ha='center',
                       size=8, weight='bold', ax=ax)
    ax.set_title(f'{team} Pass Network')
    plt.show()