#!/usr/bin/perl -w

# Ze zadaneho souboru vybere vsechna slova (kazde slovo na jednom radku,
# komentar zacina znakem #). Odstrani duplicitni polozky. Tiskne vystup
# vhodny pro Snowball skript, tzn. jednotliva slova uzavrena v jednoduchych
# uvozovkach a oddelena od sebe mezerou. Jednotlive radky vystupu se pokousi
# formatovat tak, aby jejich delka nepresahla zadanou mez (pokud neni v perlu
# korektne nastavena prace s ceskymi znaky, nebude umet spravne pocitat
# jejich delku).
#
# vstup: napr. soubory scripts/prefixes.txt nebo scripts/suffixes.txt
# priklad vystupu (levy okraj nastaven na 15 mezer):
#                'in' 'ina' 'ino' 'itý' 'ivý' 'ièký' 'inký'...
)

my $progname = $0;

# na vstupu ocekava jediny parametr: vstupni soubor
if (scalar @ARGV != 1) {
  print "Usage: $progname <file with afixes>\n";
  exit 1;
}

$file = shift(@ARGV);                 #nazev souboru
$line = "";                           #jeden radek vstupniho souboru
$leftMargin = "                ";     #znaky na levem okraji vystupu
$maxWidth = 55 + length($leftMargin); #maximalni sirka jedno radku vystupu
$count = 0;                           #pocet unikatnich afixu
@afixes = ();
$prev = "";
$tmp = $leftMargin;

open(DESCFILE, $file) or die "Nelze otevrit soubor.";

while($line = <DESCFILE>) {
  next if $line =~ m/^\s*#/;  #odstraneni komentaru zacinajicich #
  next if $line =~ m/^\s*$/;  #odstraneni znaku "konec radku"
  @afixes = (@afixes, substr($line, 0, -1));  #nacteni afixu z daneho radku
}

close DESCFILE;


@afixes = sort @afixes; #serazeni pole podle abecedy (duplicitni polozky tedy budou za sebou)

foreach $afix (@afixes) {
  if ($afix ne $prev)
  { #preskakovani duplicitnich polozek
    $tmp .= "'" . $afix . "' ";
    $count = $count + 1;
  }
  $prev = $afix;
  
  if (length($tmp) > $maxWidth)
  { #pokud vystup presahne mezni delku, vlozi se novy radek
    print $tmp . "\n";
    $tmp = $leftMargin;
  }
}
print $tmp . "\n";


print "\n" . $count . "\n"; #pocet unikatnich afixu


