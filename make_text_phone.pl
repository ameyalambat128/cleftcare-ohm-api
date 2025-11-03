#!/usr/bin/env perl
# make_text_phone.pl
# Usage: make_text_phone.pl text lexicon1.txt [lexicon2.txt ...] > text-phone

use strict;
use warnings;

my ($text, @lexicons) = @ARGV;
@lexicons >= 1 or die "Usage: make_text_phone.pl text lexicon1.txt [lexicon2.txt ...] > text-phone\n";

my %lex = ();
foreach my $lexicon (@lexicons) {
    open(my $lx_fh, "<", $lexicon) or die "Could not open $lexicon\n";
    while (<$lx_fh>) {
        chomp;
        next if $_ =~ /^\s*$/;
        my @parts = split;
        next if @parts < 2;

        my $word = shift @parts;
        if (@parts >= 1 && $parts[0] eq $word) {
            shift @parts;
        }

        next if @parts == 0;
        $lex{$word} //= join(" ", @parts);
    }
    close($lx_fh);
}

open(TX, "<", $text) or die "Could not open $text\n";
while (<TX>) {
    chomp;
    my ($uttid, @words) = split;
    for (my $i = 0; $i < @words; $i++) {
        my $wid = $uttid . "." . $i;
        if (exists $lex{$words[$i]}) {
            print "$wid $lex{$words[$i]}\n";
        } else {
            warn "Word $words[$i] not in provided lexicons, substituting silence.\n";
            print "$wid sil_S\n";
        }
    }
}
close(TX);
